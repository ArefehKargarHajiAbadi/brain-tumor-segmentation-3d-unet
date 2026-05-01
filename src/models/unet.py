"""
2D U-Net implementation for brain tumor segmentation.
This model takes RGB images (3 channels) and outputs a binary segmentation mask (1 channel).
"""

import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(Conv2D -> BatchNorm -> ReLU) x 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet2D(nn.Module):
    """2D U-Net with encoder-decoder architecture and skip connections."""
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        """
        Args:
            in_channels: Number of input channels (3 for RGB).
            out_channels: Number of output channels (1 for binary mask).
            features: List of feature maps sizes at each encoder level.
        """
        super().__init__()
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder path (contracting)
        for f in features:
            self.encoder.append(DoubleConv(in_channels, f))
            in_channels = f

        # Bottleneck (deepest part)
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder path (expanding)
        self.upconvs = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2))
            self.decoder.append(DoubleConv(f * 2, f))

        # Final output layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder forward
        for enc in self.encoder:
            x = enc(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        # Decoder forward with skip connections
        for i, (upconv, dec) in enumerate(zip(self.upconvs, self.decoder)):
            x = upconv(x)
            # Handle potential size mismatch due to odd dimensions
            skip = skip_connections[-i - 1]
            if x.shape != skip.shape:
                x = torch.nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        # Output with sigmoid activation for binary segmentation
        return torch.sigmoid(self.final_conv(x))
    
if __name__ == "__main__":
    # Quick test
    model = UNet2D(in_channels=3, out_channels=1)
    x = torch.randn(1, 3, 256, 256)  # Batch=1, Channels=3, Height=256, Width=256
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print("Model works!")