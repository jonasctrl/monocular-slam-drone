import cv2
import torch
from transformers import pipeline
from PIL import Image
import numpy as np

# MODELS:
default_model_type = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"
#default_model_type = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"

class DepthAnythingEstimatorModule:
    def __init__(self, model_type=default_model_type):
        self.model_type = model_type

        self.model = None
        self.device = None
        self.pipe = None

        self.setup_model()

    def setup_model(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.pipe = pipeline(task="depth-estimation", model=self.model_type, device=self.device)

    @torch.no_grad()
    def generate_depth_map(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        try:
            result = self.pipe(pil_image)
            depth_map = result["depth"]
        except Exception as e:
            print(f"Error in depth estimation: {str(e)}")
            return None

        depth_map = np.array(depth_map, dtype=np.float32)

        return depth_map

