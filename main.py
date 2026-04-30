"""
Main training script for 2D U-Net on BraTS dataset (JPG version).
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.models.unet import UNet2D
from src.data.dataset import BraTS2DDataset

def dice_loss(pred, target, smooth=1e-6):
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return 1 - dice

def train():
    data_root = "data/BraTS21-ImageMask-Dataset"
    batch_size = 1        
    epochs = 2             
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = BraTS2DDataset(data_root, split='train')
    valid_dataset = BraTS2DDataset(data_root, split='valid')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = UNet2D(in_channels=3, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in loop:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            preds = model(images)
            loss = dice_loss(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                preds = model(images)
                loss = dice_loss(preds, masks)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(valid_loader)

        scheduler.step()
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        # ذخیره checkpoint
        torch.save(model.state_dict(), f"unet_epoch_{epoch+1}.pth")

    print("آموزش تمام شد!")

if __name__ == "__main__":
    train()