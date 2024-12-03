#!/usr/bin/env python3
import airsim
import os
import numpy as np
import cv2
import time

# Connect to AirSim
client = airsim.MultirotorClient()
client.confirmConnection()

# Camera settings
camera_name = "front-center"
rgb_image_type = airsim.ImageType.Scene
depth_image_type = airsim.ImageType.DepthPlanar

# Output directory
output_dir = "./data-nq"
os.makedirs(output_dir, exist_ok=True)

# Counter for naming files
counter = 0

try:
    while True:
        # Fetch RGB and depth images
        responses = client.simGetImages([
            airsim.ImageRequest(camera_name, rgb_image_type, pixels_as_float=False, compress=False),
            airsim.ImageRequest(camera_name, depth_image_type, pixels_as_float=True, compress=False)
        ])

        # Handle RGB image
        rgb_response = responses[0]
        if rgb_response.width > 0 and rgb_response.height > 0:
            rgb_data = np.frombuffer(rgb_response.image_data_uint8, dtype=np.uint8)
            rgb_img = rgb_data.reshape(rgb_response.height, rgb_response.width, 3)

            rgb_file_path = os.path.join(output_dir, f"rgb_image_{counter:04d}.png")
            cv2.imwrite(rgb_file_path, rgb_img)
            print(f"RGB image saved to {rgb_file_path}")

        # Handle Depth image
        depth_response = responses[1]
        if depth_response.width > 0 and depth_response.height > 0:
            # Convert depth data to a NumPy array
            depth_data = np.array(depth_response.image_data_float, dtype=np.float32).reshape(depth_response.height, depth_response.width)

            # Clip the depth values to a valid range (e.g., 0-100 meters)
            depth_data = np.clip(depth_data, 0, 100)

            # Normalize depth data to 0-255 for visualization
            depth_img = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            depth_file_path = os.path.join(output_dir, f"depth_image_{counter:04d}.png")
            cv2.imwrite(depth_file_path, depth_img)
            print(f"Depth image saved to {depth_file_path}")

        # Increment counter for unique filenames
        counter += 1

        # Optional: Add a delay between frames (e.g., 1 second)

except KeyboardInterrupt:
    print("Script terminated by user.")
