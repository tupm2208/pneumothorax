import cv2
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset
import albumentations as albu
from albumentations.pytorch.transforms import ToTensor
import random
# transform_path = "../configures/valid_transforms_1024_old.json"
# transform = albu.load(transform_path) 


class SIIMDataset(Dataset):
    def __init__(self, folder, img_size=512, type='train'):
        self.root = folder
        self.to_tensor = ToTensor()

        df = pd.read_csv('../dataset/train_folds_5.csv')
        image_name_mask = df[df['exist_labels'] == 1]['fname'].to_list()
        image_name_nomask = df[df['exist_labels'] == 0]['fname'].to_list()

        self.image_name_list = image_name_mask + image_name_nomask[:len(image_name_mask)]
        random.shuffle(self.image_name_list)
        length = len(self.image_name_list)
        num_train = int(0.8*length)
        if type == 'train':
            self.image_name_list = self.image_name_list[:num_train]
        else:
            self.image_name_list = self.image_name_list[num_train:]
        self.img_size = img_size

        print("number of sample: ", self.__len__())

    def __getitem__(self, idx):
        image_id = self.image_name_list[idx]

        size = self.img_size
        
        image_path = os.path.join(self.root, 'train' ,image_id)
        mask_path = os.path.join(self.root, 'mask2' ,image_id)
        image = cv2.imread(image_path)
        
        label = 1.0 if os.path.isfile(mask_path) else 0.0

        image = cv2.resize(image, (size, size))

        sample = {"image": image, "label": label}
        # sample = transform(**sample)
        sample = self.to_tensor(**sample)
        image = sample['image']
        label = sample['label']
        
        return image, label

    def __len__(self):
        return len(self.image_name_list)


if __name__ == '__main__':
    dataset = SIIMDataset('../dataset')

    print(next(iter(dataset)))
    pass