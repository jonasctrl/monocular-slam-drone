import rospy
from geometry_msgs.msg import Point, Quaternion
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Bool, Header
from airsim_package.msg import Pcd2WithPose
import airsim
import numpy as np
import math
import sensor_msgs.point_cloud2 as pc2

STRIDE_X = 2
STRIDE_Y = 2

MAX_DEPTH = 30

HEIGHT, WIDTH = 144, 256
CAMERA = "fc"
RATE = 10

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

        fov = camera_info.fov
        position = self.ned_to_enu_position(camera_info.pose.position)
        orientation = self.ned_to_enu_orientation(camera_info.pose.orientation)

        return fov, position, orientation

    def rotate_points_x_axis(self, points, angle_degrees=90):
        """
        Rotate points around the X-axis by the specified angle.

        Args:
            points: Nx3 array of points (x, y, z)
            angle_degrees: Rotation angle in degrees
        Returns:
            Rotated points as Nx3 array
        """
        angle_rad = np.radians(angle_degrees)

        rot_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(angle_rad), -np.sin(angle_rad)],
            [0, np.sin(angle_rad), np.cos(angle_rad)]
        ])

        return np.dot(points, rot_matrix.T)

    def publish_point_cloud_with_pose(self):
        try:
            fov, position, orientation = self.get_camera_info()

            response = self.client.simGetImages(
                [airsim.ImageRequest(CAMERA, airsim.ImageType.DepthPlanar, pixels_as_float=True, compress=False)]
            )[0]

            rospy.loginfo(f"Depth data width: {response.width}, height: {response.height}")
            #rospy.loginfo(f"Camera fov: {fov}")

            depth_data = np.array(response.image_data_float, dtype=np.float32).reshape(response.height, response.width)

            image_width = response.width
            image_height = response.height

            # Compute fx and fy from FOV
            hfov_rad = fov * math.pi / 180.0
            fx = (image_width / 2.0) / math.tan(hfov_rad / 2.0)
            fy = fx

            # Compute principal point coordinates
            cx = image_width / 2.0
            cy = image_height / 2.0

            # Generate grid of pixel coordinates with strides
            u_coords = np.arange(0, image_width, STRIDE_X)
            v_coords = np.arange(0, image_height, STRIDE_Y)
            u_grid, v_grid = np.meshgrid(u_coords, v_coords)

            # Flatten the arrays
            u_flat = u_grid.flatten()
            v_flat = v_grid.flatten()
            z_flat = depth_data[v_flat.astype(int), u_flat.astype(int)]

            # Filter valid depth values
            valid = (z_flat > 0) & (z_flat < MAX_DEPTH)
            u_valid = u_flat[valid]
            v_valid = v_flat[valid]
            z_valid = z_flat[valid]

            # Compute x, y coordinates
            x = (u_valid - cx) * z_valid / fx
            y = (v_valid - cy) * z_valid / fy


            points = np.vstack((x, y, z_valid)).transpose()
            points_rotated = self.rotate_points_x_axis(points)

            # Create the PointCloud2 message
            fields = [
                PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1)
            ]

            header = Header()
            header.seq = self.sequence
            header.stamp = rospy.Time.now()
            header.frame_id = 'cam'
            point_cloud_msg = pc2.create_cloud(header, fields, points_rotated)

            # Create Pcd2WithPose message
            pcd2_with_pose_msg = Pcd2WithPose()
            pcd2_with_pose_msg.pcd = point_cloud_msg
            pcd2_with_pose_msg.position = position
            pcd2_with_pose_msg.orientation = orientation
            pcd2_with_pose_msg.is_global_frame = Bool(data=False)

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
