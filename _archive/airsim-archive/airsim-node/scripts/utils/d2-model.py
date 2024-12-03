import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from dpt.models import DPTDepthModel
from PIL import Image
import numpy as np
from glob import glob
import os


# Dataset
class DepthDataset(Dataset):
    def __init__(self, rgb_dir, depth_dir, transform=None):
        self.rgb_images = sorted(glob(os.path.join(rgb_dir, "*")))
        self.depth_images = sorted(glob(os.path.join(depth_dir, "*")))
        self.transform = transform

    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, idx):
        rgb_image = Image.open(self.rgb_images[idx]).convert("RGB")
        depth_image = Image.open(self.depth_images[idx]).convert("L")
        depth_image = np.array(depth_image, dtype=np.float32) / 255.0  # Normalize depth

        if self.transform:
            rgb_image = self.transform(rgb_image)

        depth_image = torch.from_numpy(depth_image).unsqueeze(0)  # Add channel dimension
        return rgb_image, depth_image


# Transformation for input images
transform = transforms.Compose([
    transforms.Resize((160, 256)),  # Resize to dimensions divisible by 32
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = DepthDataset(rgb_dir="./data/rgb", depth_dir="./data/depth", transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Load the DPT model
model = DPTDepthModel(
    path=None,  # Pretrained weights, if available
    backbone="vitb_rn50_384",  # Hybrid backbone with ResNet + ViT
    non_negative=True,
)

# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss and optimizer
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for rgb_images, depth_images in dataloader:
        rgb_images = rgb_images.to(device)
        depth_images = depth_images.to(device)

        optimizer.zero_grad()
        outputs = model(rgb_images)  # Predict depth

        # Rescale output depth map to match target dimensions
        outputs_rescaled = F.interpolate(outputs, size=(144, 256), mode="bilinear", align_corners=False)

        loss = criterion(outputs_rescaled, depth_images)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Save the fine-tuned model
torch.save(model.state_dict(), "dpt_finetuned_rescaled.pth")
