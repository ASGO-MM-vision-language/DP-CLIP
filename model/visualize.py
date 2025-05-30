
import numpy as np
import cv2
import os
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
# from scipy.misc import imread

import pdb

def vis_img(images, file_names, rand_index=None, save_path='./vis', max_idx=1):
    im_mean = [0.485, 0.456, 0.406]
    im_std = [0.229, 0.224, 0.225]
    images = images.cpu().numpy()
    batch_size = images.shape[0]
    for i in range(batch_size):
        if i >= max_idx:
            break
        image = images[i]
        im = image.transpose(1,2,0)
        height, width = im.shape[0], im.shape[1]

        background = np.ones((height,width,3)) * 255

        im = im * np.array(im_std).astype(float)
        im = im + np.array(im_mean)
        im = im * 255.
        im = np.clip(im,0,255)

        # im = cv2.resize(im, (128, height))
        im = im[:,:,::-1]
        background = im  # [10:height+10, 20:128+20]

        file_name = file_names[i].split('/')[-1]
        if rand_index is None:
            com_file_name = os.path.join(save_path, file_name)
        else:
            com_file_name = os.path.join(save_path, file_name.split('.')[0] + '-' + str(rand_index) + '.jpg') 
        print(i, com_file_name)
        cv2.imwrite(com_file_name, background)
     
    return


def vis_rank(sk_path, pred_img_paths, save_path, if_correspond=None, name_ex=None):
    height = 224
    width = 224
    background = np.ones((height + 20,width * 11 + 20, 3)) * 255

    query_img = cv2.imread(sk_path)
    query_img = cv2.resize(query_img, (height, width))

    background[10:height+10, :width, :] = query_img

    i = 0
    for img_path in pred_img_paths:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (height, width))
        # if i < 5:
        h = 10 
        w = (i+1)*width+10*(i+1)
        # else:
        #     h = height+20
        #     w = (i-4)*width+10*(i-4)

        background[10:height+10, w:w+width, :] = img
        # print(i, h, w)

        if if_correspond is not None and if_correspond[i]:
            # if if_correspond[i] == False:
            color = (0, 0, 255)  ## BGR
            start_point = (w, h)
            end_point = (w+width, h+height)
            thickness = 3
            background = cv2.rectangle(background, start_point, end_point, color, thickness)
        
        i += 1

    if if_correspond is not None and np.array(if_correspond).sum() == 0:
        category = save_path.split('/')[-1]
        img_path_name = sk_path.split(os.path.sep)[-1].split('-')[0] + '.jpg'
        correspond_img_path = os.path.join('../datasets/Sketchy/', 'photo_basic', category, img_path_name)
        print('correspond_img_path:', correspond_img_path, sk_path)

        cop_img = cv2.imread(correspond_img_path)
        cop_img = cv2.resize(cop_img, (height, width))

        background[height+20:height*2 + 20, :width, :] = cop_img

        
    query_name = sk_path.split('/')[-1]
    if name_ex is not None:
        query_name = query_name.split('.')[0] + name_ex + '.' + query_name.split('.')[1]
    # print('save ranking list for', sk_path)
    cv2.imwrite(os.path.join(save_path, query_name), background)
    
    return
