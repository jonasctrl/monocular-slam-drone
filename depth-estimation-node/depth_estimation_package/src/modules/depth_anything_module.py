import cv2
import torch
from PIL import Image
import numpy as np
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
import torchvision.transforms as transforms

class DepthAnythingEstimatorModule:
    def __init__(self, model_path="depth_anything_finetuned.pth"):
        self.model_path = model_path
        self.model = None
        self.device = None
        self.transform = None
        self.setup_model()

    def setup_model(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Initialize base model
        self.model = DepthAnything.from_pretrained("LiheYoung/depth_anything_vits14")

        # Load your fine-tuned weights
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Setup transform pipeline
        self.transform = transforms.Compose([
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

    @torch.no_grad()
    def generate_depth_map(self, frame):
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Apply transformations
            rgb_input = {"image": frame_rgb}
            rgb_transformed = self.transform(rgb_input)["image"]
            input_tensor = torch.from_numpy(rgb_transformed).unsqueeze(0).to(self.device)

            # Get depth prediction
            depth_map = self.model(input_tensor).squeeze().cpu().numpy()

            # Optional: normalize to similar range as original model
            depth_map = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)

            return depth_map

        except Exception as e:
            print(f"Error in depth estimation: {str(e)}")
            return None

