
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F

from .triplet_loss import TripletLoss

class Loss():
    def __init__(self, opts):

        self.local = opts.local
        self.local_loss_weight = opts.local_loss_weight
        self.distance_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)
        
        self.loss_fn = nn.TripletMarginWithDistanceLoss(distance_function=self.distance_fn, margin=opts.margin)
        self.hard = opts.hard
        self.hard_triplet_loss = TripletLoss(margin=opts.margin, if_hard=True)
 


    def calculate(self, sk_feat, img_feat, neg_feat, category_label, FG_label, batch_idx, current_epoch, visual_prompt=None):
        if self.local:
            sk_feat_local, img_feat_local, neg_feat_local = sk_feat[:, 1:], img_feat[:, 1:], neg_feat[:, 1:]
            sk_feat, img_feat, neg_feat = sk_feat[:, 0], img_feat[:, 0], neg_feat[:, 0]
            loss_triplet_ori_list, loss_triplet_hard_list = [], []
            # print(sk_feat_local.shape, sk_feat.shape)

        loss_triplet_hard, loss_triplet_ori = torch.zeros(1).cuda(), torch.zeros(1).cuda()
        
        if self.hard:
            weight_hard_trip = int(current_epoch / 10) * 0.25
            if batch_idx == 0:
                print('weight_hard_trip:', weight_hard_trip)
            if weight_hard_trip < 1:
                loss_triplet_ori_global = self.loss_fn(sk_feat, img_feat, neg_feat)
                loss_triplet_ori += loss_triplet_ori_global
                
                if self.local:
                    loss_triplet_ori_list.append(loss_triplet_ori_global.item())
                    for i in range(sk_feat_local.shape[1]):
                        loss_triplet_ori_local = self.loss_fn(sk_feat_local[:,i], img_feat_local[:,i], neg_feat_local[:,i])
                        loss_triplet_ori += self.local_loss_weight * loss_triplet_ori_local
                        loss_triplet_ori_list.append(loss_triplet_ori_local.item())

            if weight_hard_trip > 0:
                loss_triplet_hard_global = self.hard_triplet_loss(sk_feat, img_feat, labels=FG_label)
                loss_triplet_hard += loss_triplet_hard_global
                
                if self.local:
                    loss_triplet_hard_list.append(loss_triplet_hard_global.item())
                    for i in range(sk_feat_local.shape[1]):
                        loss_triplet_hard_local = self.hard_triplet_loss(sk_feat_local[:,i], img_feat_local[:,i], labels=FG_label)
                        loss_triplet_hard += self.local_loss_weight * loss_triplet_hard_local
                        loss_triplet_hard_list.append(loss_triplet_hard_local.item())

            loss_triplet = (1-weight_hard_trip) * loss_triplet_ori + weight_hard_trip * loss_triplet_hard
                
        else:
            loss_triplet = self.loss_fn(sk_feat, img_feat, neg_feat)


        if batch_idx % 50 == 0:
            print('training: Epoch:', current_epoch, ' idx:', batch_idx, 'triplet loss:', '%.4f' % loss_triplet.item())
                # '%.4f' % loss_triplet_ori.item(), '%.4f' % loss_triplet_hard.item() )
        
        return loss_triplet