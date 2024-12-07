import sys
import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from typing import Tuple

# NOTE: Dynamically add monodepth2 to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
monodepth2_dir = os.path.join(current_dir, "monodepth2")
sys.path.append(monodepth2_dir)

from monodepth2.networks import ResnetEncoder, DepthDecoder
from monodepth2.layers import disp_to_depth

base_path = os.path.dirname(os.path.abspath(__file__))
default_model_type = "mono_640x192"

class MonoDepth2DepthEstimatorModule:
    def __init__(self, model_type: str = default_model_type):
        """Initialize the MonoDepth2 module with optimizations."""
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.encoder = None
        self.depth_decoder = None
        self.feed_width = None
        self.feed_height = None

        self.transform = transforms.ToTensor()
        
        self._interpolation_mode = "bilinear"
        self._align_corners = False
        
        self._setup_monodepth2()
        
        # NOTE: Enable inference mode optimizations
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)

    def _setup_monodepth2(self) -> None:
        """Setup MonoDepth2 models with optimizations."""
        model_path = os.path.join(base_path, "models", self.model_type)
        encoder_path = os.path.join(model_path, "encoder.pth")
        depth_decoder_path = os.path.join(model_path, "depth.pth")

        self.encoder = ResnetEncoder(18, False)
        loaded_dict_enc = torch.load(encoder_path, map_location=self.device)

        self.feed_height = loaded_dict_enc['height']
        self.feed_width = loaded_dict_enc['width']

        filtered_dict_enc = {
            k: v for k, v in loaded_dict_enc.items() 
            if k in self.encoder.state_dict()
        }
        self.encoder.load_state_dict(filtered_dict_enc)
        self.encoder.to(self.device)
        self.encoder.eval()

        self.depth_decoder = DepthDecoder(
            num_ch_enc=self.encoder.num_ch_enc,
            scales=range(4)
        )
        self.depth_decoder.load_state_dict(
            torch.load(depth_decoder_path, map_location=self.device)
        )
        self.depth_decoder.to(self.device)
        self.depth_decoder.eval()

        try:
            self.encoder = torch.jit.script(self.encoder)
            self.depth_decoder = torch.jit.script(self.depth_decoder)
        except Exception:
            pass 

    @torch.inference_mode()  # More efficient than @torch.no_grad()
    def generate_depth_map(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Generate depth map with optimized processing."""
        original_height, original_width = frame_rgb.shape[:2]

        input_image = cv2.resize(
            frame_rgb, 
            (self.feed_width, self.feed_height),
            interpolation=cv2.INTER_LINEAR
        )

        input_image = self.transform(input_image).unsqueeze_(0).to(
            self.device, non_blocking=True
        )

        features = self.encoder(input_image)
        outputs = self.depth_decoder(features)
        disp = outputs[("disp", 0)]

        disp_resized = torch.nn.functional.interpolate(
            disp,
            (original_height, original_width),
            mode=self._interpolation_mode,
            align_corners=self._align_corners
        ).squeeze_()

        depth_map = disp_resized.cpu().numpy()
        cv2.normalize(
            depth_map,
            depth_map,
            0,
            1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F
        )
        
        return (depth_map * 255).astype(np.uint8)

    def __call__(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Convenient call method for depth estimation."""
        return self.generate_depth_map(frame_rgb)
