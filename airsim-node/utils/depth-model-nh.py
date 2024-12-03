import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np

# Define Dataset
class DepthDataset(Dataset):
    def __init__(self, rgb_dir, depth_dir, transform=None):
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.transform = transform
        self.rgb_files = sorted(os.listdir(rgb_dir))
        self.depth_files = sorted(os.listdir(depth_dir))

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        rgb_path = os.path.join(self.rgb_dir, self.rgb_files[idx])
        depth_path = os.path.join(self.depth_dir, self.depth_files[idx])

        rgb_image = Image.open(rgb_path).convert("RGB")
        depth_image = Image.open(depth_path)

        if self.transform:
            rgb_image = self.transform(rgb_image)

        depth_image = torch.tensor(
            np.array(depth_image, dtype=np.float32) / 255.0
        ).unsqueeze(0)

        return rgb_image, depth_image


# Paths
rgb_path = './data/rgb'
depth_path = './data/depth'

# Data Preparation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
dataset = DepthDataset(rgb_dir=rgb_path, depth_dir=depth_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)


# Model Definition (Example CNN-based depth estimator)
class SimpleDepthModel(nn.Module):
    def __init__(self):
        super(SimpleDepthModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Initialize Model, Loss, and Optimizer
model = SimpleDepthModel()
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=1e-4)

# Training Loop
for epoch in range(10):  # Number of epochs
    model.train()
    epoch_loss = 0
    for rgb, depth in dataloader:
        optimizer.zero_grad()
        output = model(rgb)
        loss = criterion(output, depth)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch + 1}/{10}, Loss: {epoch_loss / len(dataloader)}")

# Save the model
torch.save(model.state_dict(), "./depth_model.pth")
