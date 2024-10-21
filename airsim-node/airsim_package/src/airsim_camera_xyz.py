import rospy
from geometry_msgs.msg import Point, Quaternion
from cv_bridge import CvBridge, CvBridgeError
import airsim
import numpy as np
from airsim_package.msg import RGBWithPose, DepthWithPose

CAMERA = "front-center"
RATE = 10

class AirSimDatafeedNode:

    def __init__(self):
        rospy.init_node('airsim_data_publisher')

        self.client = airsim.MultirotorClient(ip="host.docker.internal", port=41451)
        self.client.confirmConnection()

        self.rgb_pub = rospy.Publisher('/rgb_with_pose', RGBWithPose, queue_size=1)
        self.depth_pub = rospy.Publisher('/ground_truth/depth_with_pose', DepthWithPose, queue_size=1)

        self.bridge = CvBridge()
        self.sequence = 0

    def get_camera_info(self):
        camera_info = self.client.simGetCameraInfo(CAMERA)
        position = Point()
        orientation = Quaternion()

        position.x = camera_info.pose.position.x_val
        position.y = camera_info.pose.position.y_val
        position.z = camera_info.pose.position.z_val

        orientation.x = camera_info.pose.orientation.x_val
        orientation.y = camera_info.pose.orientation.y_val
        orientation.z = camera_info.pose.orientation.z_val
        orientation.w = camera_info.pose.orientation.w_val

        return position, orientation
    
    def publish_rgb_with_pose(self):
        try:
            response = self.client.simGetImages([airsim.ImageRequest(CAMERA, airsim.ImageType.Scene, pixels_as_float=False, compress=False)])[0]
            img_rgb = np.frombuffer(response.image_data_uint8, dtype=np.uint8).reshape(response.height, response.width, 3)

            # Convert RGB image to ROS Image message
            rgb_image_msg = self.bridge.cv2_to_imgmsg(img_rgb, encoding="bgr8")
            rgb_image_msg.header.seq = self.sequence
            rgb_image_msg.header.stamp = rospy.Time.now()
            rgb_image_msg.header.frame_id = "camera"

            # Get camera position and orientation
            position, orientation = self.get_camera_info()

            # Create and publish RGBWithPose message
            rgb_with_pose_msg = RGBWithPose()
            rgb_with_pose_msg.rgb_image = rgb_image_msg
            rgb_with_pose_msg.position = position
            rgb_with_pose_msg.orientation = orientation

            self.rgb_pub.publish(rgb_with_pose_msg)

        except CvBridgeError as e:
            rospy.logerr("CvBridge error: %s", e)

    def publish_depth_with_pose(self):
        try:
            response = self.client.simGetImages(
                [airsim.ImageRequest(CAMERA, airsim.ImageType.DepthPlanar, pixels_as_float=True, compress=False)])[0]

            # Ensure the depth map is in float32 format (32FC1)
            img_depth = np.array(response.image_data_float, dtype=np.float32).reshape(response.height, response.width)

            # Convert depth image to ROS Image message
            depth_image_msg = self.bridge.cv2_to_imgmsg(img_depth, encoding="32FC1")
            depth_image_msg.header.seq = self.sequence
            depth_image_msg.header.stamp = rospy.Time.now()
            depth_image_msg.header.frame_id = "camera"

            # Get camera position and orientation
            position, orientation = self.get_camera_info()

            # Create and publish DepthWithPose message
            depth_with_pose_msg = DepthWithPose()
            depth_with_pose_msg.depth_image = depth_image_msg
            depth_with_pose_msg.position = position
            depth_with_pose_msg.orientation = orientation

            self.depth_pub.publish(depth_with_pose_msg)

        except CvBridgeError as e:
            rospy.logerr(f"CvBridge error: {str(e)}")

        except Exception as e:
            rospy.logerr(f"Error in publish_depth_with_pose: {str(e)}")

    def run(self):
        rate = rospy.Rate(RATE)

        while not rospy.is_shutdown():
            self.publish_rgb_with_pose()
            self.publish_depth_with_pose()

            self.sequence += 1
            rate.sleep()

if __name__ == '__main__':
    node = AirSimDatafeedNode()
    node.run()
