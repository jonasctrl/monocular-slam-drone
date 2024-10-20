import cv2
import numpy as np
import torch
import torch.hub

default_model_type = "DPT_Large"


class MiDaSDepthEstimatorModule:
    def __init__(self, model_type=default_model_type):
        self.model_type = model_type
        
        self.midas = None
        self.device = None
        self.transform = None
        
        self.setup_midas()

    def setup_midas(self):
        self.midas = torch.hub.load("intel-isl/MiDaS", self.model_type)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.midas.to(self.device)
        self.midas.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if self.model_type == "DPT_Large" or self.model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

    @torch.no_grad()
    def generate_depth_map(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(frame_rgb).to(self.device)

        prediction = self.midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth_map = prediction.cpu().numpy()
        depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
        return (depth_map * 255).astype(np.uint8)
