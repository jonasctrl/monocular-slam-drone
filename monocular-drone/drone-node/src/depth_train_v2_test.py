import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import glob
import warnings
from depth_anything_v2.dpt import DepthAnythingV2
from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet

warnings.filterwarnings('ignore', message='xFormers')
warnings.filterwarnings('ignore', category=UserWarning)


class ImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = glob.glob(os.path.join(data_dir, "*_*_Scene-*.png"))
        print(f"Found {len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]

        # Load and process image
        rgb_image = cv2.imread(img_path)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        rgb_image = rgb_image.astype(np.float32) / 255.0

        # Store original image properly
        original_image = cv2.imread(img_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        if self.transform:
            rgb_image = {"image": rgb_image}
            rgb_image = self.transform(rgb_image)["image"]

        return rgb_image, original_image, img_path


def load_model():
    model = DepthAnythingV2(
        encoder='vits',
        features=64,
        out_channels=[48, 96, 192, 384],
        use_bn=False,
        use_clstoken=False
    )

    weights_path = "depth_anything_v2_metric_vkitti_vits.pth"
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Pretrained weights not found at {weights_path}")

    state_dict = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state_dict)
    print("Loaded pretrained weights successfully")
    return model


def process_images(data_dir, output_dir="depth_outputs"):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Set up transforms
    transform = transforms.Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method="minimal",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    # Create dataset and dataloader
    dataset = ImageDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # Set up model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model()
    model.to(device)
    model.eval()

    print("Processing images...")
    with torch.no_grad():
        for rgb, original, img_path in tqdm(dataloader):
            # Move input to device
            rgb = rgb.to(device)

            # Generate depth map
            depth = model(rgb)

            # Convert depth to numpy and normalize to 0-255
            depth_map = depth.squeeze().cpu().numpy()
            depth_map = ((depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255).astype(np.uint8)

            # Create color map
            depth_colormap = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)

            # Get original image dimensions and convert to proper format for display
            original_img = original[0].numpy()  # Remove batch dimension
            original_height, original_width = original_img.shape[0:2]

            # Resize depth maps to match original image size
            depth_resized = cv2.resize(depth_map, (original_width, original_height))
            depth_color_resized = cv2.resize(depth_colormap, (original_width, original_height))

            # Save outputs
            base_name = os.path.splitext(os.path.basename(img_path[0]))[0]
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_depth_gray.png"), depth_resized)
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_depth_color.png"),
                        cv2.cvtColor(depth_color_resized, cv2.COLOR_RGB2BGR))

            # Convert RGB to BGR for display
            original_display = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)

            # Display images
            cv2.imshow('Original', original_display)
            cv2.imshow('Depth Map', depth_resized)
            cv2.imshow('Depth Colormap', depth_color_resized)

            key = cv2.waitKey(1)
            if key == 27:  # ESC key
                break

    cv2.destroyAllWindows()
    print(f"Processing complete! Results saved in {output_dir}")


if __name__ == "__main__":
    data_dir = "./data-nh"  # Change this to your dataset directory
    process_images(data_dir)