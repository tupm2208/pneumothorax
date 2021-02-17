import cv2
import numpy as np
import os

path = 'dataset/mask2'

image_names = os.listdir(path)

for image_name in image_names:
    image = cv2.imread(f'dataset/train/{image_name}')
    mask = cv2.imread(f'dataset/mask2/{image_name}')

    size = 700
    image = cv2.resize(image, (size, size))
    mask = cv2.resize(mask, (size, size))
    mask[mask[:,:,0]!=0] = (0,255,255)

    dst = cv2.addWeighted(image,0.95,mask,0.05,0)
    
    cv2.imshow('dst',dst)
    cv2.imshow('src', image)
    q = cv2.waitKey(0)
    if q == 27:
        cv2.destroyAllWindows()
        break
    