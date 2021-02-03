import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import pydicom
from PIL import Image
import os
from multiprocessing import Pool
from tqdm import tqdm

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

inputdir = 'datasets/input/stage_2_images'
outputdir = 'datasets/output'

test_list = [os.path.basename(e) for e in glob.glob(f'{inputdir}/*.dcm')]

def action(name):
    ds = pydicom.read_file(os.path.join(inputdir, name))
    img = ds.pixel_array
    img_mem = Image.fromarray(img)
    img_mem.save(os.path.join(outputdir, name.replace('.dcm','.png')))

pool = Pool(10)

# for _ in tqdm(pool.imap_unordered(action, test_list)):
#     pass
print(len(test_list))
# for name in test_list[:1]:
#     ds = pydicom.read_file(os.path.join(inputdir, name))
#     img = ds.pixel_array
#     img_mem = Image.fromarray(img)
#     img_mem.save(os.path.join(outputdir, name.replace('.dcm','.png')))