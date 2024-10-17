#!/usr/bin/env python3
import airsim
import rospy
import numpy as np
from sensor_msgs.msg import Image, Imu
from cv_bridge import CvBridge
import time

TARGET_FPS = 20
CAMERA = "front-center"

class AirSimDataPublisher:
    def __init__(self):
        rospy.init_node('airsim_data_publisher', anonymous=True)

        self.client = airsim.MultirotorClient(ip="host.docker.internal", port=41451)
        self.client.confirmConnection()

        self.image_pub = rospy.Publisher('/cam0/image_raw', Image, queue_size=1)
        self.imu_pub = rospy.Publisher('/imu0', Imu, queue_size=1)

        self.bridge = CvBridge()
        self.image_type = airsim.ImageType.Scene

        self.start_time = time.time()

        self.seq = 0
        self.frame_count = 0

    def publish_image(self):
        response = self.client.simGetImages([
            airsim.ImageRequest(CAMERA, self.image_type, pixels_as_float=False, compress=False)
        ])[0]

        if response.width == 0:
            rospy.logwarn("Failed to get valid image from AirSim camera")
            return

        img_rgb = np.frombuffer(response.image_data_uint8, dtype=np.uint8).reshape(response.height, response.width, 3)
        
        img_msg = self.bridge.cv2_to_imgmsg(img_rgb, encoding="bgr8")
        img_msg.header.stamp = rospy.Time.now()
        img_msg.header.frame_id = "camera"
        img_msg.header.seq = self.seq
       
        self.image_pub.publish(img_msg)

    def publish_imu(self):
        imu_data = self.client.getImuData()
    
        imu_msg = Imu()
        imu_msg.header.seq = self.seq
        imu_msg.header.stamp = rospy.Time.now()
        imu_msg.header.frame_id = "camera"
        
        imu_msg.linear_acceleration.x = imu_data.linear_acceleration.x_val
        imu_msg.linear_acceleration.y = imu_data.linear_acceleration.y_val
        imu_msg.linear_acceleration.z = imu_data.linear_acceleration.z_val
        
        imu_msg.angular_velocity.x = imu_data.angular_velocity.x_val
        imu_msg.angular_velocity.y = imu_data.angular_velocity.y_val
        imu_msg.angular_velocity.z = imu_data.angular_velocity.z_val
        
        imu_msg.orientation.w = imu_data.orientation.w_val
        imu_msg.orientation.x = imu_data.orientation.x_val
        imu_msg.orientation.y = imu_data.orientation.y_val
        imu_msg.orientation.z = imu_data.orientation.z_val
        
        imu_msg.orientation_covariance = [-1] * 9
        imu_msg.angular_velocity_covariance = [-1] * 9
        imu_msg.linear_acceleration_covariance = [-1] * 9
        
        print(imu_msg)
    
        self.imu_pub.publish(imu_msg)

    def run(self):
        rate = rospy.Rate(TARGET_FPS)
        
        while not rospy.is_shutdown():
            loop_start = time.time()

            self.publish_image()
            self.publish_imu()
            
            self.frame_count += 1
            self.seq += 1

            elapsed_time = time.time() - self.start_time
            current_fps = self.frame_count / elapsed_time

            print(
                f"Iteration {self.seq}:"
                f" Current FPS: {current_fps:.2f}"
                f" Target FPS: {TARGET_FPS}"
            )

            processing_time = time.time() - loop_start
            if processing_time < 1.0 / TARGET_FPS:
                rate.sleep()
            else:
                rospy.logwarn(f"Frame {self.seq} took {processing_time:.4f}s, exceeding target frame time of {1.0/TARGET_FPS:.4f}s")

if __name__ == '__main__':
    try:
        publisher = AirSimDataPublisher()
        publisher.run()
    except rospy.ROSInterruptException:
        pass
