import sys

sys.path.append('src')

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

# from dataloaders import SIIMDataset
from models import AlbuNet
from MaskBinarizers import TripletMaskBinarization
from classify.dataloader import SIIMDataset
import numpy as np
import cv2
from tqdm import tqdm

from efficientnet_pytorch import EfficientNet
from sklearn.metrics import classification_report
from losses import ComboLoss, soft_dice_loss

# model



device = torch.device('cuda:0')
seg_model = AlbuNet().to(device)
seg_model.load_state_dict(torch.load("checkpoints/seg-9.pth"))
seg_model.eval()

classify_model = EfficientNet.from_pretrained('efficientnet-b2', num_classes=1)
classify_model = classify_model.to(device)
classify_model.load_state_dict(torch.load("/home/tupm/HDD/projects/FCN_python/pneumothorax/checkpoints/efficientb2-2.pkl"))
classify_model.eval()


valloader = DataLoader(
    SIIMDataset("dataset", type="val"),
    batch_size=1,
    num_workers=1,
    pin_memory=True,
    shuffle=True,
)

dices = []
p_dices = []
n_dices = []
threshold = 0.3
# triplets = [[0.3, 100, 0.3], [0.75, 1000, 0.4], [0.75, 2000, 0.3], [0.75, 2000, 0.4], [0.6, 2000, 0.3], [0.6, 2000, 0.4], [0.6, 3000, 0.3], [0.6, 3000, 0.4]]
triplets = [[0.5, 100, 0.3]]
binarizer_fn = TripletMaskBinarization(triplets)

for inputs, labels, mask, original in tqdm(valloader):
    inputs = inputs.to(device)
    pred = classify_model(inputs)
    p = pred.to('cpu').detach().numpy().ravel()[0]
    
    if p > 0.5:
        original = original.to(device).type(torch.float32)
        pred_mask = seg_model(original)
        # pred_mask = torch.where(pred_mask > threshold, 1.0, 0)
        pred_mask = torch.sigmoid(pred_mask).to('cpu')
        
    else:
        pred_mask = torch.zeros((1, 1, 512, 512))
    
    corrected_mask = binarizer_fn.transform(pred_mask)
    dicess = [1 - soft_dice_loss(e.type(torch.float32), mask, per_image=True) for e in corrected_mask]
    # print(pred_mask)
    dice = 1 - soft_dice_loss(pred_mask,  mask)
    dice = dice.detach()
    
    dicess.append(dice)
    # print(dicess)
    dice = max(dicess)
    if p > 0.5:
        p_dices.append(dice)
    else:
        n_dices.append(dice)
    dices.append(dice)
    
    # if p > 0.5:
    #     break

print('dices:', np.mean(dices), len(dices))
print('p_dices:', np.mean(p_dices), len(p_dices))
print('n_dices:', np.mean(n_dices), len(n_dices))
