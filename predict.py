import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from src.models.unet import UNet2D
from src.data.dataset import BraTS2DDataset

# بارگذاری مدل ذخیره شده
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet2D(in_channels=3, out_channels=1).to(device)
model.load_state_dict(torch.load("unet_epoch_2.pth", map_location=device))
model.eval()

# انتخاب یک نمونه از دیتاست تست
test_dataset = BraTS2DDataset("data/BraTS21-ImageMask-Dataset", split='test', resize=(256,256), limit=5)
sample = test_dataset[0]
image = sample['image'].unsqueeze(0).to(device)  # اضافه کردن بعد batch
true_mask = sample['mask'].squeeze().numpy()

# پیش‌بینی
with torch.no_grad():
    pred_mask = model(image).squeeze().cpu().numpy()
    pred_mask = (pred_mask > 0.5).astype(np.float32)

# نمایش تصویر ورودی، ماسک واقعی و ماسک پیش‌بینی شده
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.imshow(image[0].cpu().permute(1,2,0).numpy())
plt.title("Input MRI")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(true_mask, cmap='gray')
plt.title("True Mask")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(pred_mask, cmap='gray')
plt.title("Predicted Mask")
plt.axis('off')

plt.show()