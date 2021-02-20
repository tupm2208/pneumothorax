import cv2
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset
import albumentations as albu
from albumentations.pytorch.transforms import ToTensor

transform_path = "../configures/train_transforms_complex_512.json"
transform = albu.load(transform_path) 

class SIIMDataset(Dataset):
    def __init__(self, folder, img_size=512):
        self.root = folder
        self.to_tensor = ToTensor()

        df = pd.read_csv('../dataset/train_folds_5.csv')
        self.image_name_list = df[df['exist_labels'] == 1]['fname'].to_list()
        self.img_size = img_size

        print("number of sample: ", self.__len__())

    def __getitem__(self, idx):
        image_id = self.image_name_list[idx]

        size = self.img_size
        
        image_path = os.path.join(self.root, 'train' ,image_id)
        mask_path = os.path.join(self.root, 'mask2' ,image_id)
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, 0)

        image = cv2.resize(image, (size, size))
        mask = cv2.resize(mask, (size, size))

        sample = {"image": image, "mask": mask}
        sample = transform(**sample)
        sample = self.to_tensor(**sample)
        image = sample['image']
        mask = sample['mask']
        
        return image, mask

    def __len__(self):
        return len(self.image_name_list)


