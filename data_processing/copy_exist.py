import shutil
import os
import numpy as np
import pandas as pd


df = pd.read_csv('../dataset/train_folds_5.csv')

df = df[df['exist_labels'] != 0]

os.makedirs('../dataset/mask2', exist_ok=True)

for image_name in df['fname']:
    shutil.copy(f'../dataset/mask/{image_name}', f'../dataset/mask2/{image_name}')

print("done!")