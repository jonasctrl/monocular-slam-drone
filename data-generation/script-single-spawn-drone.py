import airsim
import os
import random
import numpy as np
import cv2
import time
import json

def saveImage(responses, idx, camera, z):
    # with open(f"{saveLocation}/{idx}_{camera}_Scene{z}.png", "wb") as f:
    #     f.write(responses[0].image_data_uint8)
    depth_data = np.array(responses[1].image_data_float, dtype=np.float32).reshape(responses[1].height, responses[1].width)
    depth_data = np.clip(depth_data, 0, 100)
    depth_img = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


# # Connect to AirSim
client = airsim.MultirotorClient()
client.confirmConnection()
print("Connected to AirSim!")


# Start drone operation
client.enableApiControl(True)
start_time = time.time()
takeoff_task = client.takeoffAsync()
takeoff_task.join()  # Wait until takeoff is complete

position = airsim.Vector3r(20, 30, -3)
orientation = airsim.to_quaternion(0, 0, 0)  # Default orientation (roll, pitch, yaw)
pose = airsim.Pose(position, orientation)

# # Teleport the drone to the new position
client.simSetVehiclePose(pose, ignore_collision=True)

# Stabilize the drone
client.moveByVelocityAsync(0, 0, 0, 1).join()
hover_task = client.hoverAsync()
hover_task.join()

print(f"Capturing images at Coordinate {5} for {5}...")
responses = client.simGetImages([
    airsim.ImageRequest("fc", airsim.ImageType.Scene),
    airsim.ImageRequest("dfc", airsim.ImageType.DepthPlanar, True)
])
saveImage(responses, 1, "Front", 5)

