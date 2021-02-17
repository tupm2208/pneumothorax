import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import albumentations as albu 

from dataloaders import SIIMDataset
from losses import ComboLoss, soft_dice_loss
from models import AlbuNet
from MaskBinarizers import TripletMaskBinarization
import numpy as np
import cv2


device = torch.device('cuda:0')
model = AlbuNet().to(device)
# checkpoint_path = "checkpoints/albunet_1024_fold0.pth"
# model.load_state_dict(torch.load(checkpoint_path))


trainloader = DataLoader(
    SIIMDataset("dataset"),
    batch_size=1,
    num_workers=2,
    pin_memory=True,
    shuffle=True,
)

optimizer = optim.Adam(model.parameters(), lr=5e-2)
criterion = ComboLoss({
    'bce': 3,
    'dice': 1,
    'focal': 4
})

for idx, batch in enumerate(trainloader):
    images, masks = batch
    images = images.to(device).type(torch.float32)
    out = model(images)
    out = torch.sigmoid(out) 
    ms = np.reshape(out.cpu().detach().numpy(), (1024, 1024))
    ms = np.where(ms > 0.5, 1, 0).astype(np.float32)
    cv2.imshow("", ms)
    cv2.imshow("original", np.reshape(masks.detach().numpy(), (1024, 1024)))
    cv2.waitKey(0)

num_epochs = 10
grad_accum = 1
triplets = [[0.5, 100, 0.3], [0.75, 1000, 0.4], [0.75, 2000, 0.3], [0.75, 2000, 0.4], [0.6, 2000, 0.3], [0.6, 2000, 0.4], [0.6, 3000, 0.3], [0.6, 3000, 0.4]]
binarizer_fn = TripletMaskBinarization(triplets)
for epoch in range(num_epochs):
    for itr, batch in enumerate(trainloader):
        images, targets = batch
        images = images.to(device).type(torch.float32)
        masks = targets.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        # optimizer.step()
        # optimizer.zero_grad()

        if itr % grad_accum == grad_accum - 1:
            optimizer.step()
            optimizer.zero_grad()

        outputs = outputs.detach().cpu()
        # meter.update(targets, outputs)
        if itr % 1 == 0:
            outputs = torch.sigmoid(outputs)
            corrected_mask = binarizer_fn.transform(outputs)

            dices = [1 - soft_dice_loss(e.type(torch.float32), targets, per_image=True) for e in corrected_mask]
            dices.append(1-soft_dice_loss(outputs, targets))
            print(dices)
            print(f"epoch: {epoch+1} |step: {itr} | loss: {loss.item()} | dice score: {max(dices)}")