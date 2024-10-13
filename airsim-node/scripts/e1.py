#!/usr/bin/env python3
import airsim
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def main():
    rospy.init_node('airsim_image_publisher', anonymous=True)
    image_pub = rospy.Publisher('/cam0/image_raw', Image, queue_size=1)
    bridge = CvBridge()

    client = airsim.MultirotorClient(ip="host.docker.internal", port=41451)
    client.confirmConnection()

    camera_name = "front-center"
    image_type = airsim.ImageType.Scene

    rate = rospy.Rate(20)
    seq = 0

    try:
        while not rospy.is_shutdown():
            # Capture image
            response = client.simGetImages([airsim.ImageRequest(camera_name, image_type, pixels_as_float=False, compress=False)])[0]

            if response.width != 0:
                # Convert to numpy array and then to ROS image message
                img_rgb = np.frombuffer(response.image_data_uint8, dtype=np.uint8).reshape(response.height, response.width, 3)
                img_msg = bridge.cv2_to_imgmsg(img_rgb, encoding="bgr8")
                img_msg.header.stamp = rospy.Time.now()
                img_msg.header.frame_id = "camera"
                img_msg.header.seq = seq

                print("Publishing image, seq: {}".format(seq))

                # Publish image
                image_pub.publish(img_msg)
                seq += 1
            else:
                rospy.logwarn("Failed to get valid image from AirSim")

            rate.sleep()

    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
