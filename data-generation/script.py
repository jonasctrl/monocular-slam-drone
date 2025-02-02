import airsim
import os
import random
import numpy as np
import cv2
import time
import json

def saveImage(responses, idx, camera, z):
    with open(f"{saveLocation}/{idx}_{camera}_Scene{z}.png", "wb") as f:
        f.write(responses[0].image_data_uint8)
    depth_data = np.array(responses[1].image_data_float, dtype=np.float32).reshape(responses[1].height, responses[1].width)
    depth_data = np.clip(depth_data, 0, 100)
    depth_img = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite(f"{saveLocation}/{idx}_{camera}_Depth{z}.png", depth_img)

# Output directory for images
output_dir = "airsim_images"

# Connect to AirSim
client = airsim.MultirotorClient()
client.confirmConnection()
print("Connected to AirSim!")

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Parameters
num_coordinates = 209  # Number of coordinates per z-value
x_range = (-250, 250)  # Map boundaries for x
y_range = (-250, 250)  # Map boundaries for y
z_values = [-5, -3, -7, -1, -9]  # Fixed z values
random_seed = 42
saveLocation = "airsim_images"
coordinates_file = "coordinates.json"

# Generate random coordinates grouped by z-value
random.seed(random_seed)
coordinates_by_z = {
    f"z={z}": [
        {"x": random.uniform(*x_range), "y": random.uniform(*y_range), "z": z}
        for _ in range(num_coordinates)
    ]
    for z in z_values
}

# Save coordinates to JSON
with open(coordinates_file, "w") as f:
    json.dump(coordinates_by_z, f, indent=4)
print(f"Coordinates saved to {coordinates_file}")

# Load coordinates from JSON
with open(coordinates_file, "r") as f:
    coordinates_by_z = json.load(f)

# Start drone operation
client.enableApiControl(True)
start_time = time.time()
takeoff_task = client.takeoffAsync()
takeoff_task.join()  # Wait until takeoff is complete


idx_start_time = time.time()
x, y, z = 10, 10, -5

position = airsim.Vector3r(x, y, z)
orientation = airsim.to_quaternion(0, 0, 0)  # Default orientation (roll, pitch, yaw)
pose = airsim.Pose(position, orientation)

# Teleport the drone to the new position
client.simSetVehiclePose(pose, ignore_collision=True)

# Stabilize the drone
client.moveByVelocityAsync(0, 0, 0, 1).join()
hover_task = client.hoverAsync()
hover_task.join()

responses = client.simGetImages([
    airsim.ImageRequest("fc", airsim.ImageType.Scene),
    airsim.ImageRequest("fc", airsim.ImageType.DepthPlanar, True)
])
saveImage(responses, 1000, "Front", z)

responses = client.simGetImages([
    airsim.ImageRequest("bc", airsim.ImageType.Scene),
    airsim.ImageRequest("bc", airsim.ImageType.DepthPlanar, True)
])
saveImage(responses, 1000, "Back", z)

responses = client.simGetImages([
    airsim.ImageRequest("lc", airsim.ImageType.Scene),
    airsim.ImageRequest("lc", airsim.ImageType.DepthPlanar, True)
])
saveImage(responses, 1000, "Left", z)

responses = client.simGetImages([
    airsim.ImageRequest("rc", airsim.ImageType.Scene),
    airsim.ImageRequest("rc", airsim.ImageType.DepthPlanar, True)
])
saveImage(responses, 1000, "Right", z)

responses = client.simGetImages([
    airsim.ImageRequest("fcr", airsim.ImageType.Scene),
    airsim.ImageRequest("fcr", airsim.ImageType.DepthPlanar, True)
])
saveImage(responses, 1000, "FrontRight", z)

responses = client.simGetImages([
    airsim.ImageRequest("fcl", airsim.ImageType.Scene),
    airsim.ImageRequest("fcl", airsim.ImageType.DepthPlanar, True)
])
saveImage(responses, 1000, "FrontLeft", z)

responses = client.simGetImages([
    airsim.ImageRequest("bcr", airsim.ImageType.Scene),
    airsim.ImageRequest("bcr", airsim.ImageType.DepthPlanar, True)
])
saveImage(responses, 1000, "BackRight", z)

responses = client.simGetImages([
    airsim.ImageRequest("bcl", airsim.ImageType.Scene),
    airsim.ImageRequest("bcl", airsim.ImageType.DepthPlanar, True)
])
saveImage(responses, 1000, "BackLeft", z)

idx_end_time = time.time()

end_time = time.time()
print(f"Total time for all coordinates: {end_time - start_time:.4f} seconds")
