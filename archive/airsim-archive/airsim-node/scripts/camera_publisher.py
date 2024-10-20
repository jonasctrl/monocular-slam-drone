#!/usr/bin/env python3
import airsim
import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

CAMERA = "front-center"
IMAGE_RATE = 15 

class CameraPublisher:
    def __init__(self):
        rospy.init_node('camera_data_publisher', anonymous=True)

        self.client = airsim.MultirotorClient(ip="host.docker.internal", port=41451)
        self.client.confirmConnection()

        self.image_pub = rospy.Publisher('/cam0/image_raw', Image, queue_size=5)

        self.bridge = CvBridge()
        self.image_type = airsim.ImageType.Scene

        self.seq = 0

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
        self.seq += 1

    def run(self):
        rate = rospy.Rate(IMAGE_RATE)
        while not rospy.is_shutdown():
            self.publish_image()
            rate.sleep()

if __name__ == '__main__':
    try:
        camera_publisher = CameraPublisher()
        camera_publisher.run()
    except rospy.ROSInterruptException:
        pass
