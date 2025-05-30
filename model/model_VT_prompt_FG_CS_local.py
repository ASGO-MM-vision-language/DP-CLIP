import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout
# from torchmetrics.functional import retrieval_average_precision,retrieval_precision
import pytorch_lightning as pl
import time 
import pickle
import os
import copy
import random
import pdb

from .clip import clip
# from .clip.weight_init import weights_init_kaiming
from exp.options import opts
from .functions import *
from .visualize import vis_img
from .make_loss import Loss
from .metrics import calculateAccuracy,cal_acc
from .get_prompt import Prompt


class Model_FG_CS_VT_prompt_local(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.opts = opts
        self.lora = opts.lora
        self.lora_scale = opts.lora_scale
        self.local = opts.local

        self.clip, _ = clip.load('ViT-B/32', device=self.device, opts=opts)

        if self.local:
            block = self.clip.visual.transformer.resblocks[-1]
            layer_norm = self.clip.visual.ln_post
            proj = self.clip.visual.proj
            self.local_trans_layer = copy.deepcopy(block)

            self.local_layer_norm = nn.ModuleList([copy.deepcopy(layer_norm) for i in range(self.local)])
            self.local_proj = nn.Parameter(
                torch.stack([copy.deepcopy(proj) for i in range(self.local)]))
            self.local_trans_layer.apply(freeze_all_but_bn)

        ## freeze 
        self.clip.visual.transformer.apply(freeze_all_but_bn)
        self.clip.visual.conv1.apply(freeze_all_but_bn)
        self.clip.transformer.apply(freeze_all_but_bn)
        freeze_all_but_bn(self.clip.token_embedding)

        for i in range(12):
            self.clip.visual.transformer.resblocks[i].attn.in_proj_weight.requires_grad_(False)
            self.clip.visual.transformer.resblocks[i].attn.in_proj_bias.requires_grad_(False)


        ## Prompt Engineering
        self.prompt_type = opts.prompt
        patch_size = self.clip.visual.conv1.weight.data.shape[-1]  ## 32
        width = self.clip.visual.conv1.weight.data.shape[0]
        if 'CS' in self.prompt_type:
            self.CS_visual_prompt_fn = Prompt(opts, patch_size, width).cuda()
        
        if 'common' in self.prompt_type:
            self.prompt_embeddings = Prompt(opts, patch_size, width).cuda() 
        
        self.category_prompts = {}
        ## the category label feature is obtained with: get_text_prompt.py
        self.category_text_dict = pickle.load(open('../Weights/Sketchy_all_classes_text.pkl', 'rb'))


        self.distance_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)
        
        ## Loss Function
        self.loss = Loss(opts)

        self.best_metric = -1e3
        self.cal_acc = calculateAccuracy(opts.local)

        self.vis = opts.vis
        self.vis_path = opts.vis_path
        if self.vis and not os.path.exists(self.vis_path):
            os.mkdir(self.vis_path)

        self.vis_rank = opts.vis_rank
        self.vis_rank_path = opts.vis_rank_path
        if self.vis_rank and not os.path.exists(self.vis_rank_path):
            os.mkdir(self.vis_rank_path)
        
        self.random_sample = [random.randint(0, opts.batch_size-1), random.randint(0, opts.batch_size-1), 
                              random.randint(0, opts.batch_size-1), random.randint(0, opts.batch_size-1)]
        print('self.random_sample :', self.random_sample)



    def configure_optimizers(self):
        
        clip_parameters = self.clip.parameters()
        clip_parameters_id = list(map(id, clip_parameters))
        other_parameters = list(filter(lambda p: id(p) not in clip_parameters_id, self.parameters()))
        
        print('ori parameters:', len(clip_parameters_id), len(other_parameters))
        # for i in range(len(other_parameters)):
        #     print(i, other_parameters[i].shape)

        if self.lora:
            lora_parameters_id = []
            for i in range(12):
                lora_each_layer = list(map(id, self.clip.visual.transformer.resblocks[i].mlp_lora.parameters()))
                lora_parameters_id.extend(lora_each_layer)

            for i in range(12):
                lora_parameters_id.append(id(self.clip.visual.transformer.resblocks[i].attn.lora_A))
                lora_parameters_id.append(id(self.clip.visual.transformer.resblocks[i].attn.lora_B))

            if self.lora_scale:
                lora_scale_parameters_id = list(map(id, self.clip.visual.attn_lora_SS.parameters()))
                clip_parameters = list(filter(lambda p: id(p) not in lora_scale_parameters_id and id(p) not in lora_parameters_id, 
                    self.clip.parameters()))
                clip_parameters_id = list(map(id, clip_parameters))
                other_parameters = list(filter(lambda p: id(p) not in clip_parameters_id, self.parameters()))

                print('lora_scale, parameters', len(clip_parameters), len(other_parameters), len(lora_parameters_id), len(lora_scale_parameters_id))
            else:
                clip_parameters = list(filter(lambda p: id(p) not in lora_parameters_id, self.clip.parameters()))
                clip_parameters_id = list(map(id, clip_parameters))
                other_parameters = list(filter(lambda p: id(p) not in clip_parameters_id, self.parameters()))

                print('lora, parameters', len(clip_parameters), len(other_parameters), len(lora_parameters_id))
        else:
            clip_parameters = self.clip.parameters()

        optimizer = torch.optim.Adam([
        {'params': clip_parameters, 'lr': self.opts.clip_LN_lr}, 
        {'params': other_parameters, 'lr': self.opts.prompt_lr}])

        return optimizer


    def forward(self, sk_tensor, img_tensor, neg_tensor=None, category=None):
        neg_feat = None

        sk_token_feat = self.clip.visual.forward_init_token(sk_tensor)
        img_token_feat = self.clip.visual.forward_init_token(img_tensor)
        if neg_tensor is not None:
            neg_token_feat = self.clip.visual.forward_init_token(neg_tensor)

        if 'CS' in self.prompt_type:
            if neg_tensor is not None:
                # random choose 1 sketch and 2 photos
                trips = torch.cat([sk_token_feat[self.random_sample[0]], img_token_feat[self.random_sample[1]], 
                                   img_token_feat[self.random_sample[2]]], dim=0)
                CS_visual_prompt_embeddings = self.CS_visual_prompt_fn.generate_cate_adapt_prompt(trips)
            else:
                if category in self.category_prompts.keys():
                    CS_visual_prompt_embeddings = self.category_prompts[category]
                else: 
                    # trips = torch.cat([sk_token_feat[0], img_token_feat[int(img_tensor.shape[0]/2)], img_token_feat[-1]], dim=0)
                    trips = torch.cat([sk_token_feat[self.random_sample[0]], img_token_feat[self.random_sample[1]], 
                                   img_token_feat[self.random_sample[2]]], dim=0)
                    CS_visual_prompt_embeddings = self.CS_visual_prompt_fn.generate_cate_adapt_prompt(trips)
                    self.category_prompts[category] = CS_visual_prompt_embeddings
            # print('CS_visual_prompt_embeddings:', CS_visual_prompt_embeddings.shape)

        if self.prompt_type == 'CS_shallow' or self.prompt_type == 'CS_deep':
            prompt_embeddings = CS_visual_prompt_embeddings
        
        elif self.prompt_type == 'common_shallow' or self.prompt_type == 'common_deep':
            prompt_embeddings = self.prompt_embeddings.generate_common_prompt()
        else:
            prompt_embeddings = None
        
        if self.local:
            cate_text = self.category_text_dict[category]
            sk_token_feats, sk_feat_global = self.clip.encode_image(sk_token_feat, prompt_embeddings, text_embed=cate_text)
            img_token_feats, img_feat_global = self.clip.encode_image(img_token_feat, prompt_embeddings, text_embed=cate_text)

            sk_local_feats = self.get_local_feats(sk_token_feats)
            img_local_feats = self.get_local_feats(img_token_feats)

            sk_feat = torch.cat([sk_feat_global.unsqueeze(1), sk_local_feats], dim=1)
            img_feat = torch.cat([img_feat_global.unsqueeze(1), img_local_feats], dim=1)
            
            if neg_tensor is not None:
                neg_token_feats, neg_feat_global = self.clip.encode_image(neg_token_feat, prompt_embeddings, text_embed=cate_text)
                neg_local_feats = self.get_local_feats(neg_token_feats)
                neg_feat = torch.cat([neg_feat_global.unsqueeze(1), neg_local_feats], dim=1)

            # print(sk_local_feats.shape, 'sk_local_feats:', sk_feat.shape, img_feat.shape)

        else:
            cate_text = self.category_text_dict[category]
            sk_feat = self.clip.encode_image(sk_token_feat, prompt_embeddings, text_embed=cate_text)
            img_feat = self.clip.encode_image(img_token_feat, prompt_embeddings, text_embed=cate_text) 

            if neg_tensor is not None:
                neg_feat = self.clip.encode_image(neg_token_feat, prompt_embeddings, text_embed=cate_text)

        return sk_feat, img_feat, neg_feat


    def get_local_feats(self, all_token_feats):
        # print('all_token_feats:', all_token_feats.shape)  ## L * BS * dim
        cls_token = all_token_feats[0:1]
        x = all_token_feats[1:1+49]


        local_num = 5
        x_local_1 = torch.cat([x[i*7:i*7+local_num] for i in range(local_num)], dim=0)
        x_local_2 = torch.cat([x[i*7-local_num:i*7] for i in range(1, 1+local_num)], dim=0)
        x_local_3 = torch.cat([x[i*7:i*7+local_num] for i in range(7-local_num, 7)], dim=0)
        x_local_4 = torch.cat([x[i*7-local_num:i*7] for i in range(8-local_num, 8)], dim=0)
        x_local_feats = torch.stack([x_local_1, x_local_2, x_local_3, x_local_4])
        # print('x_local_feats:', x_local_feats.shape)

        local_feats_list = []

        for i in range(self.local):
            local_feat = self.local_trans_layer(torch.cat((cls_token, x_local_feats[i]), dim=0))
            local_feat = local_feat.permute(1, 0, 2) 
            local_feat = self.local_layer_norm[i](local_feat[:, 0])
            local_feat = local_feat @ self.local_proj[i]
            local_feats_list.append(local_feat)

        local_feats = torch.stack(local_feats_list, dim=1)
        # print(local_feats.shape, 'local_feats')
        return local_feats



    def training_step(self, batch, batch_idx):

        sk_tensor, img_tensor, neg_tensor, category, category_label, FG_label = batch[:6]
        
        sk_feat, img_feat, neg_feat = self.forward(sk_tensor, img_tensor, neg_tensor, category=category[0])

        loss = self.loss.calculate(sk_feat, img_feat, neg_feat, category_label, FG_label, 
                    batch_idx, self.current_epoch)

        self.log('train_loss', loss.data)
       
        if self.vis:
            sk_path, FG_img_path = batch[7:9]
            vis_img(img_tensor, FG_img_path)
            vis_img(sk_tensor, sk_path)

        return loss
    
    
    def training_epoch_end(self, training_step_outputs):

        self.category_prompts = {}
        print('Epoch: ', self.current_epoch, 'Ended. Time:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '*'*50)
        return
    

    def validation_step(self, batch, batch_idx):
        sk_tensor, img_tensor, FG_label, category = batch[:4]
        sk_feat, img_feat, _ = self.forward(sk_tensor, img_tensor, category=category[0])
        return sk_feat, img_feat, FG_label, category
        

    def validation_epoch_end(self, val_step_outputs):
        Len = len(val_step_outputs)
        print('Val: data number', Len)
        print('Epoch: ', self.current_epoch)
        if Len == 0:
            return

        query_feat_all = torch.cat([val_step_outputs[i][0] for i in range(Len)])
        gallery_feat_all = torch.cat([val_step_outputs[i][1] for i in range(Len)])
        query_label_all = np.array(sum([list(val_step_outputs[i][2]) for i in range(Len)], []))
        query_category_all = np.array(sum([list(val_step_outputs[i][3]) for i in range(Len)], []))

        acc_1, acc_5, acc_10, fuse = self.cal_acc.forward(query_feat_all, gallery_feat_all, query_label_all, query_category_all)
        print('P@1:', acc_1)
        print('P@5:', acc_5)
        print('P@10:', acc_10)
        # print('fuse:', fuse)
        
        self.log('P_1', acc_1)

        if self.global_step > 0:
            self.best_metric = self.best_metric if  (self.best_metric > acc_1) else acc_1
        
        print ('Best Acc@1: {}'.format(self.best_metric))



    def test_step(self, batch, batch_idx):

        sk_tensor, img_tensor, FG_label, category, sk_path = batch[:5]
        sk_feat, img_feat, _ = self.forward(sk_tensor, img_tensor, category=category[0])
        
        if batch_idx % 50 ==0:
            print('Test idx:', batch_idx, FG_label[0], category[0])
        
        return sk_feat, img_feat, FG_label, category, sk_path
        

    def test_epoch_end(self, test_step_outputs):
        Len = len(test_step_outputs)
        print('Epoch: ', self.current_epoch)
        print('Test: data number', Len)
        if Len == 0:
            return
        
        query_feat_all = torch.cat([test_step_outputs[i][0] for i in range(Len)])
        gallery_feat_all = torch.cat([test_step_outputs[i][1] for i in range(Len)])
        query_label_all = np.array(sum([list(test_step_outputs[i][2]) for i in range(Len)], []))
        query_category_all = np.array(sum([list(test_step_outputs[i][3]) for i in range(Len)], []))

        query_path_all = np.array(sum([list(test_step_outputs[i][4]) for i in range(Len)], []))

        acc_1, acc_5, acc_10, fuse = self.cal_acc.forward(query_feat_all, gallery_feat_all, query_label_all, 
            query_category_all, if_vis_rank=self.vis_rank, query_path_all=query_path_all, 
            vis_rank_path=self.vis_rank_path)

        print('P@1:', acc_1)
        print('P@5:', acc_5)
        print('P@10:', acc_10)
        # print('fuse:', fuse)

