import torchvision.models as models
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from dataloader import SIIMDataset
import torch.optim as optim

# model
model = models.resnet50(pretrained=True, progress=True)
model.fc = nn.Sequential(nn.Linear(2048, 128),
                                 nn.ReLU(),
                                #  nn.Dropout(0.2),
                                 nn.Linear(128, 1),
                                 nn.Sigmoid())

# data
trainloader = DataLoader(
    SIIMDataset("../dataset"),
    batch_size=2,
    num_workers=0,
    pin_memory=True,
    shuffle=True,
)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs = 1
steps = 0
running_loss = 0
print_every = 10
train_losses = []
for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        logps = logps.squeeze(1)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print(loss.item())