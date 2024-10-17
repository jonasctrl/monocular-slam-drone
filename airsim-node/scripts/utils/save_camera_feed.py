#!/usr/bin/env python3
import airsim
import os
import numpy as np
import cv2

client = airsim.MultirotorClient(ip="host.docker.internal", port=41451)
client.confirmConnection()

camera_name = "front-center"
image_type = airsim.ImageType.Scene

output_dir = "../data"
os.makedirs(output_dir, exist_ok=True)

response = client.simGetImages([airsim.ImageRequest(camera_name, image_type, pixels_as_float=False, compress=False)])[0]

if response.width > 0 and response.height > 0:
    img_data = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    img_rgb = img_data.reshape(response.height, response.width, 3)

    file_path = os.path.join(output_dir, "latest_image.png")
    cv2.imwrite(file_path, img_rgb)
    print(f"Image saved to {file_path}")
