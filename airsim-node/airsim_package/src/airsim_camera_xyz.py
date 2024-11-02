import rospy
from geometry_msgs.msg import Point, Quaternion
from airsim_package.msg import RGBWithPose
import airsim
import numpy as np
import cv2
from cv_bridge import CvBridge

CAMERA = "fc"
RATE = 10

class AirSimDatafeedNode:

    def __init__(self):
        rospy.init_node('airsim_rgb_publisher')

        self.client = airsim.MultirotorClient(ip="host.docker.internal", port=41451)
        self.client.confirmConnection()

        self.rgb_pose_pub = rospy.Publisher('/rgb_with_pose', RGBWithPose, queue_size=1)
        self.bridge = CvBridge()
        self.sequence = 0

    def ned_to_enu_position(self, position_ned):
        """
        Convert position from NED to ENU.
        Swap x and y, and negate z.
        """
        return Point(
            x=position_ned.y_val,
            y=position_ned.x_val,
            z=-position_ned.z_val
        )

    def ned_to_enu_orientation(self, orientation_ned):
        """
        Convert orientation (quaternion) from NED to ENU.
        Swap x and y, and negate z.
        """
        return Quaternion(
            x=orientation_ned.y_val,
            y=orientation_ned.x_val,
            z=-orientation_ned.z_val,
            w=orientation_ned.w_val
        )

    def get_camera_info(self):
        camera_info = self.client.simGetCameraInfo(CAMERA)
        position = self.ned_to_enu_position(camera_info.pose.position)
        orientation = self.ned_to_enu_orientation(camera_info.pose.orientation)

        return position, orientation

    def publish_rgb_with_pose(self):
        try:
            position, orientation = self.get_camera_info()

            response = self.client.simGetImages([
                airsim.ImageRequest(CAMERA, airsim.ImageType.Scene, False, False)
            ])[0]

            rospy.loginfo(f"Image width: {response.width}, height: {response.height}")

            img_1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img_bgr = img_1d.reshape(response.height, response.width, 3)

            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Convert to ROS Image message
            rgb_msg = self.bridge.cv2_to_imgmsg(img_rgb, encoding="rgb8")
            rgb_msg.header.seq = self.sequence
            rgb_msg.header.stamp = rospy.Time.now()
            rgb_msg.header.frame_id = "cam"

            # Create RGBWithPose message
            rgb_with_pose_msg = RGBWithPose()
            rgb_with_pose_msg.rgb_image = rgb_msg
            rgb_with_pose_msg.position = position
            rgb_with_pose_msg.orientation = orientation

            self.rgb_pose_pub.publish(rgb_with_pose_msg)

        except Exception as e:
            rospy.logerr(f"Error in publish_rgb_with_pose: {str(e)}")

    def run(self):
        rate = rospy.Rate(RATE)

        while not rospy.is_shutdown():
            self.publish_rgb_with_pose()
            self.sequence += 1
            rate.sleep()

if __name__ == '__main__':
    node = AirSimDatafeedNode()
    node.run()