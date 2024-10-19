#!/usr/bin/env python3
import airsim

client = airsim.MultirotorClient(ip="host.docker.internal", port=41451)
client.confirmConnection()
client.enableApiControl(True)

state = client.getMultirotorState()

imu_data = client.getImuData()
barometer_data = client.getBarometerData()
magnetometer_data = client.getMagnetometerData()
gps_data = client.getGpsData()

airsim.wait_key('Press any key to takeoff')
print("Taking off...")
client.armDisarm(True)
client.takeoffAsync().join()

while True:
    airsim.wait_key('Press any key to move vehicle to (-10, 10, -10) at 5 m/s')
    client.moveToPositionAsync(0, 0, -10, 1).join()
    
    # wait for drone to settle
    airsim.wait_key('Press any key to move vehicle to (-10, 10, -10) at 2 m/s')
    client.moveToPositionAsync(0, 0, 0, 1).join()
    