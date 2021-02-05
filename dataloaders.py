import cv2
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset
import albumentations as albu
from albumentations.pytorch.transforms import ToTensor

transform_path = "/home/tupm/HDD/projects/FCN_python/pneumothorax-segmentation/unet_pipeline/transforms/valid_transforms_1024_old.json"
transform = albu.load(transform_path) 

class SIIMDataset(Dataset):
    def __init__(self, folder):
        # self.image_name_list = os.listdir(f'{folder}/images')
        # self.image_name_list = os.listdir(f'{folder}/train')
        self.root = folder
        self.to_tensor = ToTensor()

        df = pd.read_csv('train_folds_5.csv')
        self.image_name_list = df[df['exist_labels'] == 1]['fname'].to_list()

        
        print("number of sample: ", self.__len__())
        print("number of sample: ", self.__len__())

    def __getitem__(self, idx):
        image_id = self.image_name_list[idx]
        
        image_path = os.path.join(self.root, 'train' ,image_id)
        mask_path = os.path.join(self.root, 'mask' ,image_id)
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, 0)
        # mask = np.where(cv2.imread(mask_path, 0) > 0, 1, 0).astype(np.float32)
        sample = {"image": image, "mask": mask}
        sample = transform(**sample)
        sample = self.to_tensor(**sample)
        image = sample['image']
        mask = sample['mask']
        
        # size = 1024
        
        # image = cv2.resize(image, (size, size))
        # mask = cv2.resize(mask, (size, size))
        # image = (np.reshape(image, (3, size, size)) - 127)/255.0
        # mask  = np.reshape(mask, (1, size, size))

        return image, mask

    def __len__(self):
        return len(self.image_name_list)


