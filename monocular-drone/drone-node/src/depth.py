import os
import sys
# NOTE: Dynamically add depth to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
depth_dir = os.path.join(current_dir, "depth_anything_v2")
sys.path.append(depth_dir)

from depth_anything_v2.dpt import DepthAnythingV2
from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
import torch
import cv2
import numpy as np
import torchvision.transforms as transforms


class DepthAnythingEstimatorModuleV2:
    print("DepthAnythingEstimatorModuleV2")
    def __init__(self, model_path="/catkin_ws/src/drone-node/src/depth_anything_v2_checkpoint_epoch_22.pth"):
        self.model_path = model_path
        self.model = None
        self.device = None
        self.transform = None
        self.setup_model()

    def setup_model(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.model = self.load_pretrained_model_small()

        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

    @torch.no_grad()
    def generate_depth_map(self, frame):
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = frame_rgb.astype('float32') / 255.0

            rgb_input = {"image": frame_rgb}
            rgb_transformed = self.transform(rgb_input)["image"]
            input_tensor = torch.from_numpy(rgb_transformed).unsqueeze(0).to(self.device)

            depth_map = self.model(input_tensor).squeeze().cpu().numpy()

            print(f"Depth map shape: {depth_map.shape}")
            print(f"Depth map: {depth_map}")

            # Rescale depth map to 0-255 range
            depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
            depth_map = depth_map.astype(np.uint8)

            print(f"Depth map shape: {depth_map.shape}")
            print(f"Depth map: {depth_map}")

            return depth_map

        except Exception as e:
            print(f"Error in depth estimation: {str(e)}")
            return None

    def load_pretrained_model_base(self):
        return DepthAnythingV2(
            encoder='vitb',
            features=128,
            out_channels=[96, 192, 384, 768],
            use_bn=False,
            use_clstoken=False
        )

    def load_pretrained_model_small(self):
        return DepthAnythingV2(
            encoder='vits',
            features=64,
            out_channels=[48, 96, 192, 384],
            use_bn=False,
            use_clstoken=False
        )

    def load_pretrained_model_large(self):
        return DepthAnythingV2(
            encoder='vitl',
            features=256,
            out_channels=[256, 512, 1024, 1024],
            use_bn=False,
            use_clstoken=False
        )
