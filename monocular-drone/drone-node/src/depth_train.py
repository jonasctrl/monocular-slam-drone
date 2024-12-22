import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
import cv2


class CustomDepthDataset(Dataset):
    def __init__(self, rgb_dir, depth_dir, transform=None):
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.transform = transform
        self.rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.startswith('rgb_image')])
        self.target_height = 294
        self.target_width = 518

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        rgb_file = self.rgb_files[idx]
        depth_file = rgb_file.replace('rgb_image', 'depth_image')

        rgb_path = os.path.join(self.rgb_dir, rgb_file)
        depth_path = os.path.join(self.depth_dir, depth_file)

        rgb_image = Image.open(rgb_path).convert('RGB')
        depth_image = Image.open(depth_path)

        rgb_image = np.array(rgb_image)
        depth_image = np.array(depth_image).astype(np.float32)

        if self.transform:
            rgb_image = {"image": rgb_image}
            rgb_image = self.transform(rgb_image)["image"]

            depth_image = cv2.resize(depth_image, (self.target_width, self.target_height), interpolation=cv2.INTER_NEAREST)
            depth_image = torch.from_numpy(depth_image).float()

        return rgb_image, depth_image


def train_depth_anything(rgb_dir, depth_dir, num_epochs=10, batch_size=8, learning_rate=1e-4):
    model = DepthAnything.from_pretrained("LiheYoung/depth_anything_vits14")
    model.train()

    transform = transforms.Compose([
        Resize(
            width=518,
            height=294,
            resize_target=True,
            keep_aspect_ratio=False,
            ensure_multiple_of=14,
            resize_method="minimal",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    dataset = CustomDepthDataset(rgb_dir, depth_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.L1Loss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (rgb, depth) in enumerate(dataloader):
            rgb, depth = rgb.to(device), depth.to(device)

            pred_depth = model(rgb)

            loss = criterion(pred_depth, depth)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, f'depth_anything_checkpoint_epoch_{epoch}.pth')

    torch.save(model.state_dict(), 'depth_anything_finetuned.pth')
    return model


if __name__ == "__main__":
    rgb_dir = "../data/rgb"
    depth_dir = "../data/depth"

    model = train_depth_anything(
        rgb_dir=rgb_dir,
        depth_dir=depth_dir,
        num_epochs=10,
        batch_size=8,
        learning_rate=1e-4
    )

