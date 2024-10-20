#!/usr/bin/env python3
import airsim
import numpy as np
import time

client = airsim.MultirotorClient(ip="host.docker.internal", port=41451)
client.confirmConnection()

camera_name = "front-center"
imu_name = "imu"

camera_info = client.simGetCameraInfo(camera_name)
fov_degrees = camera_info.fov

image_response = client.simGetImages([airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)])[0]
if image_response.width > 0 and image_response.height > 0:
    width = image_response.width
    height = image_response.height
    
    # Focal length calculation based on FOV and image dimensions
    fov_radians = np.deg2rad(fov_degrees)
    fx = (width / 2) / np.tan(fov_radians / 2)
    fy = (height / 2) / np.tan(fov_radians / 2)

    print("Camera Intrinsics Calculated from FOV:")
    print(f"  Focal Length X (fx): {fx}")
    print(f"  Focal Length Y (fy): {fy}")
    print(f"  Principal Point X (cx): {width / 2}")
    print(f"  Principal Point Y (cy): {height / 2}")

    print("\nImage Dimensions:")
    print(f"  Width: {width}")
    print(f"  Height: {height}")

# Retrieve IMU data sample and estimate noise characteristics
print("\nEstimating IMU Noise Characteristics...")
gyro_samples = []
acc_samples = []

for _ in range(100):
    imu_data = client.getImuData(imu_name)
    gyro_samples.append((imu_data.angular_velocity.x_val, imu_data.angular_velocity.y_val, imu_data.angular_velocity.z_val))
    acc_samples.append((imu_data.linear_acceleration.x_val, imu_data.linear_acceleration.y_val, imu_data.linear_acceleration.z_val))
    time.sleep(0.1)

# Calculate noise characteristics (standard deviation for simplicity)
gyro_noise = [np.std([s[i] for s in gyro_samples]) for i in range(3)]
acc_noise = [np.std([s[i] for s in acc_samples]) for i in range(3)]

print("Estimated IMU Noise:")
print(f"  Gyroscope Noise: {gyro_noise}")
print(f"  Accelerometer Noise: {acc_noise}")
