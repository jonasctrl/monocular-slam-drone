import rospy
from geometry_msgs.msg import Point, Quaternion
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Bool
from airsim_package.msg import Pcd2WithPose
import airsim
import numpy as np
import math

CAMERA = "front-center"
RATE = 5
HEIGHT, WIDTH = 144, 256

class AirSimDatafeedNode:

    def __init__(self):
        rospy.init_node('airsim_data_publisher')

        self.client = airsim.MultirotorClient(ip="host.docker.internal", port=41451)
        self.client.confirmConnection()

        self.cam_pcd_pose_pub = rospy.Publisher('/cam_pcd_pose', Pcd2WithPose, queue_size=1)
        self.pointcloud_pub = rospy.Publisher('/point_cloud', PointCloud2, queue_size=1)

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

    def rotate_point(self, point, axis='x', angle=90):
        """Apply rotation matrix based on the selected axis and angle."""
        x, y, z = point

        rotation_angle = math.radians(angle)

        if axis == 'x':
            # Rotate around X-axis
            rotation_matrix = np.array([[1, 0, 0],
                                        [0, math.cos(rotation_angle), -math.sin(rotation_angle)],
                                        [0, math.sin(rotation_angle), math.cos(rotation_angle)]])
        elif axis == 'y':
            # Rotate around Y-axis
            rotation_matrix = np.array([[math.cos(rotation_angle), 0, math.sin(rotation_angle)],
                                        [0, 1, 0],
                                        [-math.sin(rotation_angle), 0, math.cos(rotation_angle)]])
        else:
            # Rotate around Z-axis
            rotation_matrix = np.array([[math.cos(rotation_angle), -math.sin(rotation_angle), 0],
                                        [math.sin(rotation_angle), math.cos(rotation_angle), 0],
                                        [0, 0, 1]])

        rotated_point = np.dot(rotation_matrix, np.array([x, y, z]))
        return rotated_point

    def publish_point_cloud_with_pose(self):
        try:
            # Get camera position and orientation
            position, orientation = self.get_camera_info()

            response = self.client.simGetImages(
                [airsim.ImageRequest(CAMERA, airsim.ImageType.DepthPlanar, pixels_as_float=True, compress=False)])[0]

            depth_data = np.array(response.image_data_float, dtype=np.float32).reshape(response.height, response.width)

            fx, fy = WIDTH, HEIGHT
            cx, cy = response.width / 2, response.height / 2

            max_depth = 30
            # Compute 3D points from the depth map
            point_cloud = []
            for v in range(response.height):
                for u in range(response.width):
                    z = depth_data[v, u]
                    if z > 0 and z < max_depth:
                        x = (u - cx) * z / fx
                        y = (v - cy) * z / fy

                        point = [x, y, z]

                        rotated_point = self.rotate_point(point, axis='x', angle=90)
                        enu_point = Point(rotated_point[0], rotated_point[1], rotated_point[2])
                        point_cloud.append([enu_point.x, enu_point.y, enu_point.z])

            # Create the PointCloud2 message
            pcd_msg = PointCloud2()
            pcd_msg.header.seq = self.sequence
            pcd_msg.header.stamp = rospy.Time.now()
            pcd_msg.header.frame_id = "cam"

            # Populate point cloud fields
            pcd_msg.fields = [
                PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1)
            ]
            pcd_msg.is_bigendian = False
            pcd_msg.point_step = 12  # 3 * 4 bytes for float32 x, y, z
            pcd_msg.row_step = pcd_msg.point_step * len(point_cloud)
            pcd_msg.height = 1
            pcd_msg.width = len(point_cloud)
            pcd_msg.is_dense = True

            pcd_msg.data = np.asarray(point_cloud, np.float32).tobytes()
            #self.pointcloud_pub.publish(pcd_msg)

            # Create Pcd2WithPose message
            pcd2_with_pose_msg = Pcd2WithPose()
            pcd2_with_pose_msg.pcd = pcd_msg
            pcd2_with_pose_msg.position = position
            pcd2_with_pose_msg.orientation = orientation
            pcd2_with_pose_msg.is_global_frame = Bool(data=False)

            # Publish the message
            self.cam_pcd_pose_pub.publish(pcd2_with_pose_msg)

        except Exception as e:
            rospy.logerr(f"Error in publish_point_cloud_with_pose: {str(e)}")

    def run(self):
        rate = rospy.Rate(RATE)

        while not rospy.is_shutdown():
            self.publish_point_cloud_with_pose()

            self.sequence += 1
            rate.sleep()


if __name__ == '__main__':
    node = AirSimDatafeedNode()
    node.run()