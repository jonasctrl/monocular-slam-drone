#!/usr/bin/env python3
import airsim
import os
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp

# Load the pretrained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,  # No pre-trained weights, since we're loading custom ones
    in_channels=3,
    classes=1,
)
model.load_state_dict(torch.load("depth_model.pth", map_location=device))
model = model.to(device)
model.eval()

# Define transformations for RGB images
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# AirSim setup
client = airsim.MultirotorClient()
client.confirmConnection()

camera_name = "front-center"
rgb_image_type = airsim.ImageType.Scene

try:
    while True:
        # Get RGB image from AirSim
        responses = client.simGetImages([
            airsim.ImageRequest(camera_name, rgb_image_type, pixels_as_float=False, compress=False),
        ])
        rgb_response = responses[0]

        if rgb_response.width > 0 and rgb_response.height > 0:
            # Convert RGB response to a NumPy array
            rgb_data = np.frombuffer(rgb_response.image_data_uint8, dtype=np.uint8)
            rgb_img = rgb_data.reshape(rgb_response.height, rgb_response.width, 3)

            # Convert RGB image to a PIL image and apply transformations
            rgb_pil = transforms.ToPILImage()(rgb_img)
            input_tensor = transform(rgb_pil).unsqueeze(0).to(device)

            # Predict depth map
            with torch.no_grad():
                depth_pred = model(input_tensor).squeeze(0).squeeze(0).cpu().numpy()

            # Normalize the depth map for visualization
            depth_display = cv2.normalize(depth_pred, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            # Display the depth map
            cv2.imshow("Predicted Depth Map", depth_display)

            # Display the original RGB image
            cv2.imshow("RGB Image", rgb_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except KeyboardInterrupt:
    print("Script terminated by user.")

cv2.destroyAllWindows()
