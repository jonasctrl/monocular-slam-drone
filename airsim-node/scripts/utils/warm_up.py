#!/usr/bin/env python3
import airsim
import time

client = airsim.MultirotorClient(ip="host.docker.internal", port=41451)
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

state = client.getMultirotorState()

imu_data = client.getImuData()
barometer_data = client.getBarometerData()
magnetometer_data = client.getMagnetometerData()
gps_data = client.getGpsData()

airsim.wait_key('Press any key to takeoff')
print("Taking off...")
client.takeoffAsync().join()

# Get the initial position
home_position = client.getMultirotorState().kinematics_estimated.position
home_x = home_position.x_val
home_y = home_position.y_val
home_z = home_position.z_val

DELAY = 0.5

while True:
    # Move Up
    airsim.wait_key('Press Enter to move up and return to original position')
    client.moveToPositionAsync(home_x, home_y, home_z - 5, DELAY).join()
    client.moveToPositionAsync(home_x, home_y, home_z, DELAY).join()

    # Move Down
    # airsim.wait_key('Press Enter to move down and return to original position')
    # client.moveToPositionAsync(home_x, home_y, home_z + 5, DELAY).join()
    # client.moveToPositionAsync(home_x, home_y, home_z, DELAY).join()

    # Move Left
    airsim.wait_key('Press Enter to move left and return to original position')
    client.moveToPositionAsync(home_x, home_y - 5, home_z, DELAY).join()
    client.moveToPositionAsync(home_x, home_y, home_z, DELAY).join()

    # Move Right
    airsim.wait_key('Press Enter to move right and return to original position')
    client.moveToPositionAsync(home_x, home_y + 5, home_z, DELAY).join()
    client.moveToPositionAsync(home_x, home_y, home_z, DELAY).join()

    # Return to home position and orientation
    client.moveToPositionAsync(home_x, home_y, home_z, DELAY).join()
    client.rotateToYawAsync(0).join()  # Returns to initial yaw (facing forward)
