import torch
import torch.nn.functional as F
import os
import numpy as np
import pdb

from .visualize import vis_rank


def distance_fn(x, y):
    return 1.0 - F.cosine_similarity(x, y)



def cal_acc(query_feat_all, gallery_feat_all, query_label_all, query_category_all, if_vis_rank=False, 
            query_path_all=None, vis_rank_path=None, topk=5, name_ex=None):

    gallery_label_all = query_label_all
    gallery_category_all = query_category_all


    ## Get the non_repeated_list 
    gallery_name_non_repeated = [] 
    non_repeated_list = []
    for label in gallery_label_all:
        if label in gallery_name_non_repeated:
            non_repeated_list.append(0)
        else:
            gallery_name_non_repeated.append(label)
            non_repeated_list.append(1)
    
    non_repeated_boolTensor = torch.BoolTensor(non_repeated_list)
    ## non-repeated
    gallery_feat_all = gallery_feat_all[non_repeated_boolTensor]
    gallery_label_all = gallery_label_all[non_repeated_boolTensor]
    gallery_category_all = gallery_category_all[non_repeated_boolTensor]
    # print('query len', len(query_feat_all), len(query_label_all), len(query_category_all), 
    #         'gallery feat', len(gallery_feat_all), len(gallery_label_all), len(gallery_category_all))
        

    query_num = len(query_feat_all)
    gallery_num = len(gallery_category_all)

    gallery_category_dict = {}
    for i in range(gallery_num):
        gallery_category = gallery_category_all[i]
        if gallery_category in gallery_category_dict.keys():
            gallery_category_dict[gallery_category].append(i)
        else:
            gallery_category_dict[gallery_category] = [i]
    # print("Categories: ", len(gallery_category_dict.keys()), gallery_category_dict.keys())

    indics = torch.zeros(query_num, 100)
    for idx, query_feat in enumerate(query_feat_all):
        query_category = query_category_all[idx]
        gallery_feat_category = torch.stack([gallery_feat_all[i] for i in gallery_category_dict[query_category]])
        distance = distance_fn(query_feat.unsqueeze(0), gallery_feat_category)
        sort_indic = torch.sort(distance)[1]
        indice_each = torch.IntTensor([gallery_category_dict[query_category][i] for i in sort_indic])
        indics[idx, :len(indice_each)] = indice_each
        

    indics = indics.int()
    # print('obtain indics')

    correct_num_1, correct_num_5, correct_num_10 = 0, 0, 0
    for idx_query in range(query_num):
        pred_labels = []
        query_label = query_label_all[idx_query]
        query_category = query_category_all[idx_query]

        for idx_gal in indics[idx_query]:

            pred_labels.append(gallery_label_all[idx_gal.item()])

            if len(pred_labels) == 1:
                correct_num_1 += query_label in pred_labels
            elif len(pred_labels) == 5:
                correct_num_5 += query_label in pred_labels
            elif len(pred_labels) == 10:
                correct_num_10 += query_label in pred_labels
                break

    acc_1 = correct_num_1 / query_num
    acc_5 = correct_num_5 / query_num
    acc_10 = correct_num_10 / query_num



    if if_vis_rank:
        # vis_rank_all(indics, query_label_all, query_category_all, gallery_label_all, query_path_all, vis_rank_path, topk=7)
        save_path = './test_files'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        np.save(os.path.join(save_path, 'sketch_paths.npy'), query_path_all)
        torch.save(indics, os.path.join(save_path, 'indics_gallery.pth'))
        torch.save(query_label_all, os.path.join(save_path, 'query_label_all.pt'))
        torch.save(query_category_all, os.path.join(save_path, 'query_category_all.pt'))
        torch.save(gallery_label_all, os.path.join(save_path, 'gallery_label_all.pt'))

        for idx_query in range(query_num):
            # if idx_query < 1000:
            #     continue
            # if idx_query > 5000:
            #     break
            query_path = query_path_all[idx_query]
            print(idx_query, "query_path:", query_path)
            
            query_label = query_label_all[idx_query]
            query_category = query_category_all[idx_query]

            pred_img_paths = []
            is_correspond = []
            for idx_gal in indics[idx_query]:
                pred_label = gallery_label_all[idx_gal.item()]
                if pred_label == query_label:
                    is_correspond.append(True)
                else:
                    is_correspond.append(False)
                pred_img_path = os.path.join('../datasets/Sketchy', 'photo_basic', query_category, pred_label + '.jpg')
                pred_img_paths.append(pred_img_path)
                # print('id_gal', idx_gal, pred_img_path)
                if len(pred_img_paths) == topk:
                    break
                
            if np.array(is_correspond)[:5].sum() == 0: #1 and is_correspond[0]:
                save_path = os.path.join(vis_rank_path, query_category) #str(idx_query)+ '_' + query_name)
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                print('id_gal', idx_gal, pred_img_path)
                    
                # save_path = os.path.join(self.vis_rank_path, str(idx_query)+ '_' + query_name)
                vis_rank(query_path, pred_img_paths, save_path, is_correspond, name_ex=name_ex)


    return acc_1, acc_5, acc_10

class calculateAccuracy():
    def __init__(self, local):
        self.local = int(local)

    def cal_dis(self, query_feat_all, gallery_feat_all, query_label_all, query_category_all, if_indics=True):

        gallery_label_all = query_label_all
        gallery_category_all = query_category_all
        ## Get the non_repeated_list 
        gallery_name_non_repeated = [] 
        non_repeated_list = []
        for label in gallery_label_all:
            if label in gallery_name_non_repeated:
                non_repeated_list.append(0)
            else:
                gallery_name_non_repeated.append(label)
                non_repeated_list.append(1)
        
        non_repeated_boolTensor = torch.BoolTensor(non_repeated_list)
        ## non-repeated
        gallery_feat_all = gallery_feat_all[non_repeated_boolTensor]
        gallery_label_all = gallery_label_all[non_repeated_boolTensor]
        gallery_category_all = gallery_category_all[non_repeated_boolTensor]
        # print('query len', len(query_feat_all), len(query_label_all), len(query_category_all), 
        #         'gallery feat', len(gallery_feat_all), len(gallery_label_all), len(gallery_category_all))
        
        gallery_category_dict = {}
        for i in range(len(gallery_label_all)):
            gallery_category = gallery_category_all[i]
            if gallery_category in gallery_category_dict.keys():
                gallery_category_dict[gallery_category].append(i)
            else:
                gallery_category_dict[gallery_category] = [i]
        # print("Categories: ", len(gallery_category_dict.keys()), gallery_category_dict.keys())

        query_num = len(query_feat_all)

        indics = torch.zeros(query_num, 100)
        distance_matrix = torch.ones(query_num, 100) * 2.0

        for idx, query_feat in enumerate(query_feat_all):
            query_category = query_category_all[idx]
            gallery_feat_category = torch.stack([gallery_feat_all[i] for i in gallery_category_dict[query_category]])
            distance = distance_fn(query_feat.unsqueeze(0), gallery_feat_category)
            distance_matrix[idx, :len(distance)] = distance
            
            if if_indics:
                sort_indic = torch.sort(distance)[1]
                indice_each = torch.IntTensor([gallery_category_dict[query_category][i] for i in sort_indic])
                indics[idx, :len(indice_each)] = indice_each
        
        if if_indics:
            return indics, gallery_label_all
        else:
            return distance_matrix, gallery_category_dict, gallery_label_all

    

    def cal_acc(self, indics, query_label_all, gallery_label_all):

        indics = indics.int()

        query_num = len(query_label_all)
        correct_num_1, correct_num_5, correct_num_10 = 0, 0, 0
        if_conrresponds = []
        for idx_query in range(query_num):
            pred_labels = []
            query_label = query_label_all[idx_query]

            for idx_gal in indics[idx_query]:

                pred_labels.append(gallery_label_all[idx_gal.item()])

                if len(pred_labels) == 1:
                    correct_num_1 += query_label in pred_labels
                elif len(pred_labels) == 5:
                    correct_num_5 += query_label in pred_labels
                elif len(pred_labels) == 10:
                    correct_num_10 += query_label in pred_labels
                    break

        acc_1 = correct_num_1 / query_num
        acc_5 = correct_num_5 / query_num
        acc_10 = correct_num_10 / query_num
        return acc_1, acc_5, acc_10



    def forward(self, query_feat_all, gallery_feat_all, query_label_all, query_category_all, if_vis_rank=False, 
                query_path_all=None, vis_rank_path=None):

        # self.save_feat(query_feat_all, gallery_feat_all, query_label_all, query_category_all)

        if self.local:
            distances_list = []    

            for i in range(int(self.local+1)):
                distance, gallery_category_dict, gallery_label_all = self.cal_dis(query_feat_all[:, i], gallery_feat_all[:, i], 
                    query_label_all, query_category_all, if_indics=False)
                distances_list.append(distance)

                # indics = torch.zeros(len(query_feat_all), 100)
                # for idx in range(len(query_feat_all)):
                #     query_category = query_category_all[idx]
                #     valid_len = len(gallery_category_dict[query_category])
                #     sort_indic = torch.sort(distance[idx, :valid_len])[1]
                #     indice_each = torch.IntTensor([gallery_category_dict[query_category][i] for i in sort_indic])
                #     indics[idx, :valid_len] = indice_each

                # acc_1, acc_5, acc_10 = self.cal_acc(indics, query_label_all, gallery_label_all)
                # if i == 0:
                #     print('Global:', "acc_1, acc_5, acc_10:", acc_1, acc_5, acc_10)
                # else:
                #     print('Local:', i, "acc_1, acc_5, acc_10:", acc_1, acc_5, acc_10)


                # if if_vis_rank:
                #     self.vis_rank(indics, query_label_all, query_category_all, gallery_label_all, 
                #         query_path_all, vis_rank_path)
        

            print('-'*50)

            best_acc_1, best_acc_5, best_acc_10 = 0.0, 0.0, 0.0
            best_fuse = 0
            for fuse_idx in [2]: # range(0, 6): 
                fuse = fuse_idx / 10
                fused_distance = distances_list[0]
                for i in range(1, len(distances_list)):
                    fused_distance += fuse * distances_list[i]

                indics = torch.zeros(len(query_feat_all), 100)
                for idx in range(len(query_feat_all)):
                    query_category = query_category_all[idx]
                    valid_len = len(gallery_category_dict[query_category])
                    sort_indic = torch.sort(fused_distance[idx, :valid_len])[1]
                    indice_each = torch.IntTensor([gallery_category_dict[query_category][i] for i in sort_indic])
                    indics[idx, :valid_len] = indice_each

                acc_1, acc_5, acc_10 = self.cal_acc(indics, query_label_all, gallery_label_all)
                # print('P@1:', acc_1, 'P@5:', acc_5, 'P@10:', acc_10)
                if acc_1 > best_acc_1:
                    best_acc_1 = acc_1
                    best_acc_5 = acc_5
                    best_acc_10 = acc_10
                    best_fuse = fuse
                
            print('P@1:', best_acc_1, 'P@5:', best_acc_5, 'P@10:', best_acc_10)

            
            return best_acc_1, best_acc_5, best_acc_10, best_fuse

        else:

            indics, gallery_label_all = self.cal_dis(query_feat_all, gallery_feat_all, query_label_all, query_category_all, if_indics=True)

            acc_1, acc_5, acc_10 = self.cal_acc(indics, query_label_all, gallery_label_all)
            print('acc_1, acc_5, acc_10:', acc_1, acc_5, acc_10)

        if if_vis_rank:
            vis_rank(indics, query_label_all, query_category_all, gallery_label_all, query_path_all, vis_rank_path)
        
        return acc_1, acc_5, acc_10, 1.0



    def save_feat(self, query_feat_all, gallery_feat_all, query_label_all, query_category_all):
        save_path = './test_files'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        torch.save(query_feat_all, os.path.join(save_path, 'query_feat_all.pt'))
        torch.save(query_label_all, os.path.join(save_path, 'query_label_all.pt'))
        torch.save(query_category_all, os.path.join(save_path, 'query_category_all.pt'))
        torch.save(gallery_feat_all, os.path.join(save_path, 'gallery_feat_all.pt'))


def main():
    save_path = './test_files'
    query_feat_all = torch.load(os.path.join(save_path, 'query_feat_all.pt'))
    query_label_all = torch.load(os.path.join(save_path, 'query_label_all.pt'))
    query_category_all = torch.load(os.path.join(save_path, 'query_category_all.pt'))
    gallery_feat_all = torch.load(os.path.join(save_path, 'gallery_feat_all.pt'))
    print(query_feat_all.shape, gallery_feat_all.shape, query_label_all.shape, query_category_all.shape)


    # indics = torch.load(os.path.join(save_path, 'indics_gallery.pth'))
    # vis_rank_all(indics, query_label_all, query_category_all, gallery_label_all, query_path_all, vis_rank_path,
    # topk=7, name_ex='_g')

    cal_acc = calculateAccuracy(0)
    cal_acc.forward(query_feat_all, gallery_feat_all, query_label_all, query_category_all)

    # query_path_all = np.load(os.path.join(save_path, 'sketch_paths.npy'))
    # acc_1, acc_5, acc_10 = cal_acc(query_feat_all[:, 0], gallery_feat_all[:, 0], query_label_all, 
    #     query_category_all, if_vis_rank=True, 
    #         query_path_all=query_path_all, vis_rank_path='./vis_rank_wr5', topk=5, name_ex='_g')
    # print('acc_1, acc_5, acc_10:', acc_1, acc_5, acc_10)

    # for i in range(4):
    #     acc_1, acc_5, acc_10 = cal_acc(query_feat_all[:, i+1], gallery_feat_all[:, i+1], query_label_all, 
    #         query_category_all, if_vis_rank=True, 
    #         query_path_all=query_path_all, vis_rank_path='./vis_rank_wr5', topk=5, name_ex='_l'+str(i+1))
    #     print('Local' + str(i+1) + ' P@1:', acc_1, 'P@5:', acc_5, 'P@10:', acc_10)

    # acc_1, acc_5, acc_10 = cal_acc(query_feat_all[:, 1], gallery_feat_all[:, 1], query_label_all, 
    #     query_category_all, if_vis_rank=True, 
    #         query_path_all=query_path_all, vis_rank_path='./vis_rank', topk=7, name_ex='_l1')
    # print('acc_1, acc_5, acc_10:', acc_1, acc_5, acc_10)
# main()

