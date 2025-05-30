import os
import glob
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageOps
import random
from .transform import data_transform 
from .Sketchy_classes import unseen_classes,unseen_classes_split2

def rotate_transform(sketch, image, rotation_degree):
        angle = transforms.RandomRotation.get_params([-rotation_degree, rotation_degree])
        image = transforms.functional.rotate(image, angle, fill=255)
        sketch = transforms.functional.rotate(sketch, angle, fill=255)
        return sketch, image


class Sketchy_Basic(torch.utils.data.Dataset):

    def __init__(self, opts, transform, mode='train'):

        self.opts = opts
        self.transform = transform
        self.mode = mode

        self.train_gray = opts.train_gray
        # self.gray_transform = transforms.RandomGrayscale(p=self.train_gray)
        self.horflip = opts.horflip
        self.horflip_transform = transforms.RandomHorizontalFlip(p=1)
        self.rotation = opts.rotation
        self.rotation_degree = opts.rotation_degree


        self.all_categories = os.listdir(os.path.join(self.opts.data_dir, 'sketch'))
        if '.ipynb_checkpoints' in self.all_categories:
            self.all_categories.remove('.ipynb_checkpoints')
            
        if opts.data_split == 1:
            if self.mode == 'train':
                # self.all_categories = seen_classes
                self.all_categories = list(set(self.all_categories) - set(unseen_classes)) 
                self.seen_classes_dict = {self.all_categories[i]: i for i in range(len(self.all_categories))}
            else:
                self.all_categories = unseen_classes
        else:
            if self.mode == 'train':
                self.all_categories = list(set(self.all_categories) - set(unseen_classes_split2)) 
                self.seen_classes_dict = {self.all_categories[i]: i for i in range(len(self.all_categories))}
            else:
                self.all_categories = unseen_classes_split2


        self.all_sketches_path = []
        self.all_photos_path = {}
        self.all_FG_labels = []

        print('data split:', opts.data_split, self.mode, len(self.all_categories))
        number = 0

        for category in self.all_categories:
            self.all_sketches_path.extend(glob.glob(os.path.join(self.opts.data_dir, 'sketch', category, '*.png')))
            self.all_photos_path[category] = glob.glob(os.path.join(self.opts.data_dir, 'photo_basic', category, '*.jpg'))
            self.all_FG_labels.extend([path.split(os.path.sep)[-1].split('.jpg')[0] for path in self.all_photos_path[category]])
            
            number += len(self.all_photos_path[category])
        
        self.all_FG_labels = np.array(self.all_FG_labels)
            
        print('mode:', mode, 'number of sketches:', len(self.all_sketches_path), 
              'number in all photos:', number, 'number of labels:', len(self.all_FG_labels))
        
    def __len__(self):
        return len(self.all_sketches_path)  # self.number_sketch #
        
    def __getitem__(self, index):
        filepath = self.all_sketches_path[index]                
        category = filepath.split(os.path.sep)[-2]
        FG_label = filepath.split(os.path.sep)[-1].split('-')[0]

        sk_path  = filepath
        sk_data  = ImageOps.pad(Image.open(sk_path).convert('RGB'),  size=(self.opts.max_size, self.opts.max_size))

        ## FG samples
        FG_img_path = os.path.join(self.opts.data_dir, 'photo_basic', category, FG_label + '.jpg')
        FG_img_data = ImageOps.pad(Image.open(FG_img_path).convert('RGB'), size=(self.opts.max_size, self.opts.max_size))
        
        if self.mode == 'train':

            neg_photos = self.all_photos_path[category].copy()
            neg_photos.remove(FG_img_path)
            hard_neg_path = np.random.choice(neg_photos)
            hard_neg_data = ImageOps.pad(Image.open(hard_neg_path).convert('RGB'), size=(self.opts.max_size, self.opts.max_size))
            
            if random.random() < self.horflip:
                sk_data  = self.horflip_transform(sk_data)
                FG_img_data = self.horflip_transform(FG_img_data)

            if random.random() < self.rotation:
                sk_data, FG_img_data = rotate_transform(sk_data, FG_img_data, self.rotation_degree)

            if random.random() < self.train_gray:
                FG_img_data = transforms.functional.to_grayscale(FG_img_data, num_output_channels=3)
                hard_neg_data = transforms.functional.to_grayscale(hard_neg_data, num_output_channels=3)

            sk_tensor  = self.transform(sk_data)
            FG_img_tensor = self.transform(FG_img_data)
            hard_neg_tensor = self.transform(hard_neg_data)

            category_label = self.seen_classes_dict[category]
            FG_label = np.where(self.all_FG_labels == FG_label)[0][0]

            return (sk_tensor, FG_img_tensor, hard_neg_tensor, category, category_label, FG_label, sk_path, FG_img_path, hard_neg_path) 
                    # img_tensor, neg_tensor, img_path, neg_path) 


        else:  ## test or val
            
            sk_tensor  = self.transform(sk_data)
            FG_img_tensor = self.transform(FG_img_data)

            return (sk_tensor, FG_img_tensor, FG_label, category, sk_path, FG_img_path)
        


if __name__ == '__main__':
    from experiments.options import opts
    import tqdm

    dataset_transforms = data_transform(opts)
    dataset_train = Sketchy_Basic(opts, dataset_transforms, mode='train')
    dataset_val = Sketchy_Basic(opts, dataset_transforms, mode='val', used_cat=dataset_train.all_categories)

    idx = 0
    for data in tqdm.tqdm(dataset_val):
        # continue
        (sk_tensor, img_tensor, neg_tensor, filename,
            sk_data, img_data, neg_data) = data

        canvas = Image.new('RGB', (224*3, 224))
        offset = 0
        for im in [sk_data, img_data, neg_data]:
            canvas.paste(im, (offset, 0))
            offset += im.size[0]
        canvas.save('output/%d.jpg'%idx)
        idx += 1
