import torch
import torchvision
from dataset import CarvanaDataset
from torch.utils.data import DataLoader
from tqdm import tqdm


def save_checkpoint(state, filename='checkpoint.pth'):
    print("=> saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])


def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transforms,
    val_transforms,
    num_workers=4,
    pin_memory=True
):
    train_ds = CarvanaDataset(train_dir, train_maskdir, train_transforms)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )


    val_ds = CarvanaDataset(val_dir, val_maskdir, val_transforms)
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )

    return train_loader, val_loader


def check_accuracy(loader, model, device='cuda'):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = torch.sigmoid(model(x)).squeeze(dim=1)
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()  + 1e-7) / ((preds + y).sum() + 1e-7)
    
    print(f'got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}')
    print(f'Dice score: {dice_score/len(loader)}')
    model.train()

def save_predictions_as_imgs(loader, model, folder='saved_images/', device='cuda'):
    model.eval()
    for idx, (x, y) in tqdm(enumerate(loader)):
        x = x.to(device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
        # torchvision.utils.save_image(y.unsqueeze(1), )
    
    model.train()