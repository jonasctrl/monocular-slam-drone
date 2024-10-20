#!/usr/bin/env python3

import rospy
import cv2
from cv_bridge import CvBridge
from depth_estimation_package.msg import RGBWithPose, DepthWithPose
from modules.depth_anything_module import DepthAnythingEstimatorModule

GUI = True

class DepthEstimationNode:
    def __init__(self):
        rospy.init_node('depth_estimation_node', anonymous=True)

        self.bridge = CvBridge()
        self.depth_estimator_module = DepthAnythingEstimatorModule()

        self.image_sub = rospy.Subscriber('/rgb_with_pose', RGBWithPose, self.image_callback)
        self.depth_pub = rospy.Publisher('/depth_with_pose', DepthWithPose, queue_size=1)

        rospy.loginfo("Depth estimation node initialized.")

    def image_callback(self, msg):
        rospy.loginfo("Received RGBWithPose message")

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg.rgb_image, desired_encoding='bgr8')
            rospy.loginfo(f"Image size: {cv_image.shape}")

            position = msg.position
            orientation = msg.orientation

            depth_map = self.depth_estimator_module.generate_depth_map(cv_image)

            if GUI:
                depth_display = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                cv2.imshow("Depth Map", depth_display)
                cv2.waitKey(1)  # Allows OpenCV to process the image window

            depth_msg = self.bridge.cv2_to_imgmsg(depth_map, encoding='32FC1')

            # Create and publish DepthWithPose message
            depth_with_pose_msg = DepthWithPose()
            depth_with_pose_msg.depth_image = depth_msg
            depth_with_pose_msg.position = position
            depth_with_pose_msg.orientation = orientation

            self.depth_pub.publish(depth_with_pose_msg)
            rospy.loginfo("Published depth_with_pose message")

        except Exception as e:
            rospy.logerr(f"Error processing image: {str(e)}")

    def run(self):
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("Shutting down depth estimation node...")
        finally:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        node = DepthEstimationNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
