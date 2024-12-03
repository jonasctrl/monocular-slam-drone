#!/usr/bin/env python3
import airsim
import os
import numpy as np
import cv2
import time

client = airsim.MultirotorClient()
client.confirmConnection()

camera_name = "fc"
rgb_image_type = airsim.ImageType.Scene
depth_image_type = airsim.ImageType.DepthPlanar

output_dir = "./data_518_deg15_nh"
os.makedirs(output_dir, exist_ok=True)

rgb_dir = os.path.join(output_dir, "rgb")
os.makedirs(rgb_dir, exist_ok=True)

depth_dir = os.path.join(output_dir, "depth")
os.makedirs(depth_dir, exist_ok=True)


counter = 0
try:
    while True:
        responses = client.simGetImages([
            airsim.ImageRequest(camera_name, rgb_image_type, pixels_as_float=False, compress=False),
            airsim.ImageRequest(camera_name, depth_image_type, pixels_as_float=True, compress=False)
        ])

        rgb_response = responses[0]
        if rgb_response.width > 0 and rgb_response.height > 0:
            rgb_data = np.frombuffer(rgb_response.image_data_uint8, dtype=np.uint8)
            rgb_img = rgb_data.reshape(rgb_response.height, rgb_response.width, 3)

            rgb_file_path = os.path.join(output_dir, "rgb", f"rgb_image_{counter:04d}.png")
            cv2.imwrite(rgb_file_path, rgb_img)
            print(f"RGB image saved to {rgb_file_path}")

        depth_response = responses[1]
        if depth_response.width > 0 and depth_response.height > 0:
            depth_data = np.array(depth_response.image_data_float, dtype=np.float32).reshape(depth_response.height, depth_response.width)

            depth_data = np.clip(depth_data, 0, 100)
            depth_img = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            depth_file_path = os.path.join(output_dir, "depth", f"depth_image_{counter:04d}.png")
            cv2.imwrite(depth_file_path, depth_img)
            print(f"Depth image saved to {depth_file_path}")

        counter += 1
        time.sleep(0.1)
except KeyboardInterrupt:
    print("Script terminated by user.")

