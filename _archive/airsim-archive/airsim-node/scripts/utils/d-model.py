import os
import numpy as np
from glob import glob
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import segmentation_models_pytorch as smp


class DepthDataset(Dataset):
    def __init__(self, rgb_dir, depth_dir, transform=None):
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.transform = transform

        self.rgb_images = sorted(glob(os.path.join(rgb_dir, '*')))
        self.depth_images = sorted(glob(os.path.join(depth_dir, '*')))

        assert len(self.rgb_images) == len(self.depth_images), "Mismatch between RGB and depth images"

    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, idx):
        # Load RGB image
        rgb_path = self.rgb_images[idx]
        rgb_image = Image.open(rgb_path).convert('RGB')

        # Load depth image
        depth_path = self.depth_images[idx]
        depth_image = Image.open(depth_path).convert('L')
        depth_image = np.array(depth_image).astype(np.float32) / 255.0  # Normalize to [0, 1]
        depth_image = torch.from_numpy(depth_image)

        if self.transform:
            rgb_image = self.transform(rgb_image)

        depth_image = transforms.Resize((rgb_image.shape[1], rgb_image.shape[2]))(depth_image.unsqueeze(0))
        depth_image = depth_image.squeeze(0)

        return rgb_image, depth_image


# Transformations for RGB images
rgb_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# Create dataset and dataloader
dataset = DepthDataset(
    rgb_dir='./data/rgb',
    depth_dir='./data/depth',
    transform=rgb_transform
)

batch_size = 4
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the U-Net model with a pre-trained encoder
model = smp.Unet(
    encoder_name="resnet34",  # Encoder: ResNet34
    encoder_weights="imagenet",  # Pre-trained on ImageNet
    in_channels=3,  # RGB images
    classes=1,  # Single channel depth map
)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for rgb_images, depth_images in dataloader:
        rgb_images = rgb_images.to(device)
        depth_images = depth_images.to(device)

        optimizer.zero_grad()
        outputs = model(rgb_images)
        outputs = outputs.squeeze(1)

        loss = criterion(outputs, depth_images)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), 'depth_model.pth')
