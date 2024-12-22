import os
import sys
# NOTE: Dynamically add depth to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
depth_dir = os.path.join(current_dir, "depth_anything_v2")
sys.path.append(depth_dir)


import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2
from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
import cv2
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import glob
import warnings
import gc

warnings.filterwarnings('ignore', message='xFormers')
warnings.filterwarnings('ignore', category=UserWarning)

class CustomDepthDataset(Dataset):
    def __init__(self, data_dirs, transform=None, max_images=100000):
        self.transform = transform
        self.pairs = []
        self.target_height = 518
        self.target_width = 518
        count = 0


        for data_dir in data_dirs:
            rgb_images = glob.glob(os.path.join(data_dir, "*_*_Scene-*.png"))

            print(f"Found {len(rgb_images)} images in {data_dir}")
            for rgb_path in rgb_images:
                base_name = os.path.basename(rgb_path)
                parts = base_name.replace('.png', '').split('_')
                if len(parts) != 3:
                    print(f"Skipping {base_name} - unexpected format")
                    continue

                index, direction, scene_part = parts
                height = scene_part.split('-')[1]
                depth_name = f"{index}_{direction}_Depth-{height}.png"
                depth_path = os.path.join(data_dir, depth_name)

                if os.path.exists(depth_path):
                    self.pairs.append((rgb_path, depth_path))
                else:
                    print(f"Warning: No matching depth image found for {base_name}")

                count += 1
                if count >= max_images:
                    break

        print(f"Found {len(self.pairs)} valid image pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        rgb_path, depth_path = self.pairs[idx]
        
        try:
            # Load RGB image
            rgb_image = cv2.imread(rgb_path)
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            rgb_image = rgb_image.astype(np.float32) / 255.0

            # Load depth image
            depth_image = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
            depth_image = depth_image.astype(np.float32) / 255.0

        except Exception as e:
            print(f"Error loading images: {e}")
            return torch.zeros(3, self.target_height, self.target_width), torch.zeros(self.target_height, self.target_width)

        if self.transform:
            rgb_image = {"image": rgb_image}
            rgb_image = self.transform(rgb_image)["image"]

            depth_image = cv2.resize(depth_image, (self.target_width, self.target_height),
                                   interpolation=cv2.INTER_NEAREST)
            depth_image = torch.from_numpy(depth_image).float()

        # Debug prints for first few images
        if idx < 5:
            print(f"Image {idx} stats:")
            print(f"RGB range: [{rgb_image.min():.4f}, {rgb_image.max():.4f}]")
            print(f"Depth range: [{depth_image.min():.4f}, {depth_image.max():.4f}]")

        return rgb_image, depth_image

class BerHuLoss(nn.Module):
    def __init__(self, threshold=0.2):
        super().__init__()
        self.threshold = threshold

    def forward(self, pred, target):
        diff = torch.abs(target - pred)
        mask = (diff <= self.threshold).float()

        l1_loss = diff * mask
        l2_loss = (diff * diff + self.threshold * self.threshold) / (2 * self.threshold)
        l2_loss = l2_loss * (1 - mask)

        return torch.mean(l1_loss + l2_loss)

def load_pretrained_model():
    model = DepthAnythingV2(
        encoder='vitl',
        features=128,
        out_channels=[96, 192, 384, 768],
        use_bn=False,
        use_clstoken=False,
        #max_depth=20.0
    )

    weights_path = "depth_anything_v2_metric_vkitti_vitl.pth"
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Pretrained weights not found at {weights_path}")

    state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    print("Loaded pretrained weights successfully")

    return model

def save_checkpoint(state, filename):
    print(f"Saving checkpoint to {filename}")
    torch.save(state, filename)
    print("Checkpoint saved successfully")

def train_depth_anything(data_dirs, num_epochs=10, batch_size=8, learning_rate=1e-4):
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device_type == 'cpu':
        raise RuntimeError("CUDA is not available. Please check your GPU installation.")
    
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Load model
    try:
        model = load_pretrained_model()
    except FileNotFoundError as e:
        print(e)
        return

    model.train()

    # Set up transforms
    transform = transforms.Compose([
        Resize(
            width=518,
            height=518,
            resize_target=True,
            keep_aspect_ratio=False,
            ensure_multiple_of=14,
            resize_method="minimal",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    # Create dataset and split into train/val
    dataset = CustomDepthDataset(data_dirs, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create data loaders
    cpu_cores = os.cpu_count()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cpu_cores,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cpu_cores,
        pin_memory=True,
        persistent_workers=True
    )

    # Set up training components
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    criterion = BerHuLoss()
    scaler = GradScaler(device_type)

    device = torch.device(device_type)
    model.to(device)
    model = model.to(memory_format=torch.channels_last)

    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 10

    print("Starting training...")
    try:
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0
            train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}")

            for batch_idx, (rgb, depth) in enumerate(train_pbar):
                rgb, depth = rgb.to(device, non_blocking=True), depth.to(device, non_blocking=True)

                with autocast(device_type=device_type):
                    pred_depth = model(rgb)
                    loss = criterion(pred_depth, depth)

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()
                train_pbar.set_postfix({'loss': loss.item()})

                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()

            avg_train_loss = train_loss / len(train_loader)

            # Validation phase
            model.eval()
            val_loss = 0
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc="Validating")
                for rgb, depth in val_pbar:
                    rgb, depth = rgb.to(device, non_blocking=True), depth.to(device, non_blocking=True)

                    with autocast(device_type=device_type):
                        pred_depth = model(rgb)
                        loss = criterion(pred_depth, depth)

                    val_loss += loss.item()
                    val_pbar.set_postfix({'loss': loss.item()})

            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            scheduler.step(avg_val_loss)

            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, f'depth_anything_v2_checkpoint_epoch_{epoch}.pth')

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                print(f"New best validation loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"Validation loss didn't improve. Patience: {patience_counter}/{max_patience}")

            if patience_counter >= max_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

            torch.cuda.empty_cache()
            gc.collect()

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current model state...")
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
        }, "interrupted_depth_anything_v2_model.pth")
        print("Model saved. Exiting...")

    print("Training completed!")
    return model

if __name__ == "__main__":
    data_dirs = [
        "./data-nh",
        "./airsim_images-MS/airsim_images-MS",
    ]

    print("Starting Depth-Anything-V2 training script")
    print(f"Data directories: {data_dirs}")

    model = train_depth_anything(
        data_dirs=data_dirs,
        num_epochs=30,
        batch_size=8,
        learning_rate=1e-5
    )
