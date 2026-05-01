"""
Dataset loader for BraTS 2D JPG images.
Each sample: MRI image (RGB) and corresponding tumor mask (grayscale, binary).
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import glob
import numpy as np

class BraTS2DDataset(Dataset):
    """PyTorch Dataset for 2D Brain Tumor Segmentation (BraTS converted to JPG)."""
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir: Path to 'BraTS21-ImageMask-Dataset' folder.
            split: 'train', 'valid', or 'test'.
            transform: Optional torchvision transforms (not used here).
        """
        self.image_dir = os.path.join(root_dir, split, 'images')
        self.mask_dir = os.path.join(root_dir, split, 'masks')
        
        # Collect all image files (both .jpg and .png)
        self.images = sorted(
            glob.glob(os.path.join(self.image_dir, "*.jpg")) +
            glob.glob(os.path.join(self.image_dir, "*.png"))
        )
        self.transform = transform

    def __len__(self):
        return min(100,len(self.images))

    def __getitem__(self, idx):
        # Load image and corresponding mask (same filename)
        img_path = self.images[idx]
        mask_name = os.path.basename(img_path)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # grayscale

        # Convert to tensor and normalize to [0,1]
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(np.array(mask)).float() / 255.0
        mask = (mask > 0.5).float()  # binary threshold

        # Add channel dimension (C, H, W) -> actually mask is (H,W), we'll unsqueeze
        # Return as dictionary for clarity
        sample = {
            'image': image,      # (3, H, W)
            'mask': mask.unsqueeze(0)  # (1, H, W)
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


# Quick test to verify dataset works
if __name__ == "__main__":
    # Adjust path if needed: point to the data folder
    dataset = BraTS2DDataset(root_dir="data/BraTS21-ImageMask-Dataset", split='train')
    print(f"Number of training samples: {len(dataset)}")
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Mask shape: {sample['mask'].shape}")
    print(f"Image value range: [{sample['image'].min():.2f}, {sample['image'].max():.2f}]")
    print(f"Mask value range: [{sample['mask'].min():.2f}, {sample['mask'].max():.2f}]")
    print("Dataset works!")