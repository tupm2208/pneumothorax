import torchvision.models as models
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from dataloader import SIIMDataset
import torch.optim as optim
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm
from sklearn.metrics import classification_report

# model
model = EfficientNet.from_pretrained('efficientnet-b2', num_classes=1)

# data
trainloader = DataLoader(
    SIIMDataset("../dataset"),
    batch_size=4,
    num_workers=3,
    pin_memory=True,
    shuffle=True,
)

valloader = DataLoader(
    SIIMDataset("../dataset", type="val"),
    batch_size=4,
    num_workers=3,
    pin_memory=True,
    shuffle=True,
)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = model.to(device)
model.load_state_dict(torch.load("/home/tupm/HDD/projects/FCN_python/pneumothorax/checkpoints/efficientb2-3.pkl"))
THRESHOLD = 0.5

epochs = 3
steps = 0
running_loss = 0
print_every = 100
train_losses = []
for epoch in range(epochs):
    
    for inputs, labels in tqdm(trainloader):
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        logps = logps.squeeze(1)
        labels = labels.type(torch.float32)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        if steps % print_every == 0:
            print(loss.item())

    # model.eval()
    gr = []
    preds = []
    for inputs, labels in tqdm(valloader):
        steps += 1
        inputs = inputs.to(device)
        logps = model.forward(inputs)
        logps = logps.squeeze(1)

        gr += labels.detach().numpy().tolist()
        preds += logps.to('cpu').detach().numpy().tolist()

    
    predicts = np.where(np.array(preds)>THRESHOLD, 1.0, 0)
    print(classification_report(np.array(gr), predicts))

    torch.save(model.state_dict(), f"../checkpoints/efficientb2-{epoch+1}.pkl")
    # model.train()
    
    