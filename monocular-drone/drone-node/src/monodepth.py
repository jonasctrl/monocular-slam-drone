import sys
import os

# NOTE: Dynamically add monodepth2 to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
monodepth2_dir = os.path.join(current_dir, "monodepth2")
sys.path.append(monodepth2_dir)

import cv2
import numpy as np
import torch
from torchvision import transforms

from monodepth2.networks import ResnetEncoder, DepthDecoder
from monodepth2.layers import disp_to_depth


base_path = os.path.dirname(os.path.abspath(__file__))
default_model_type = "mono_640x192"

class MonoDepth2DepthEstimatorModule:
    def __init__(self, model_type=default_model_type):
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.encoder = None
        self.depth_decoder = None
        self.feed_width = None
        self.feed_height = None

        self.setup_monodepth2()

    def setup_monodepth2(self):
        model_path = f"{base_path}/models/{self.model_type}"
        encoder_path = f"{model_path}/encoder.pth"
        depth_decoder_path = f"{model_path}/depth.pth"

        self.encoder = ResnetEncoder(18, False)
        loaded_dict_enc = torch.load(encoder_path, map_location=self.device)

        self.feed_height = loaded_dict_enc['height']
        self.feed_width = loaded_dict_enc['width']

        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(filtered_dict_enc)
        self.encoder.to(self.device)
        self.encoder.eval()

        self.depth_decoder = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc, scales=range(4))
        loaded_dict = torch.load(depth_decoder_path, map_location=self.device)
        self.depth_decoder.load_state_dict(loaded_dict)
        self.depth_decoder.to(self.device)
        self.depth_decoder.eval()

    @torch.no_grad()
    def generate_depth_map(self, frame_rgb):
        original_height, original_width = frame_rgb.shape[:2]
        input_image = cv2.resize(frame_rgb, (self.feed_width, self.feed_height))
        input_image = transforms.ToTensor()(input_image).unsqueeze(0).to(self.device)

        features = self.encoder(input_image)
        outputs = self.depth_decoder(features)
        disp = outputs[("disp", 0)]

        disp_resized = torch.nn.functional.interpolate(
            disp, (original_height, original_width), mode="bilinear", align_corners=False
        ).squeeze()

        depth_map = disp_resized.cpu().numpy()
        depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
        # NOTE: Convert to 8-bit depth map
        depth_map = (depth_map * 255).astype(np.uint8)
        
        return depth_map
