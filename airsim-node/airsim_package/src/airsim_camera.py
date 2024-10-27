import rospy
from cv_bridge import CvBridge
import airsim
import numpy as np
from sensor_msgs.msg import Image

CAMERA = "front-center"
RATE = 20

class AirSimDatafeedNode:

    def __init__(self):
        rospy.init_node('airsim_camera_publisher')

        self.client = airsim.MultirotorClient(ip="host.docker.internal", port=41451)
        self.client.confirmConnection()

        self.image_pub = rospy.Publisher('/cam0/image_raw', Image, queue_size=1)

        self.bridge = CvBridge()
        self.sequence = 0

    def publish_rgb(self):
        response = self.client.simGetImages([
            airsim.ImageRequest(CAMERA, airsim.ImageType.Scene, pixels_as_float=False, compress=False)
        ])[0]

        if response.width == 0:
            rospy.logwarn("Failed to get valid image from AirSim camera")
            return

        img_rgb = np.frombuffer(response.image_data_uint8, dtype=np.uint8).reshape(response.height, response.width, 3)

        img_msg = self.bridge.cv2_to_imgmsg(img_rgb, encoding="bgr8")
        img_msg.header.stamp = rospy.Time.now()
        img_msg.header.frame_id = "camera"
        img_msg.header.seq = self.sequence

        self.image_pub.publish(img_msg)
        
    def run(self):
        rate = rospy.Rate(RATE)

        while not rospy.is_shutdown():
            self.publish_rgb()

            self.sequence += 1
            rate.sleep()

if __name__ == '__main__':
    node = AirSimDatafeedNode()
    node.run()
