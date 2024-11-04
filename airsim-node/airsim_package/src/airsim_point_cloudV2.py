import rospy
from geometry_msgs.msg import Point, Quaternion
from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import Path
from std_msgs.msg import Bool
from airsim_package.msg import Pcd2WithPose
import airsim
import numpy as np
import math
import random

import time
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import CubicSpline

CAMERA = "front-center"
RATE = 5
HEIGHT, WIDTH = 144, 256
PTS_MULT = 2


##############
#  pathhing  #
##############
def find_nearest_path_index(path, cur_pos):
    """
    Finds the index of the closest point in the path to the current drone position.

    Returns:
    - int: Index of the nearest point in the path.
    """
    distances = [np.linalg.norm(cur_pos - np.array(p)) for p in path]
    return np.argmin(distances)

def create_path_to_goal(current_position, goal_path):
    """Creates a path from current position to the nearest point on the goal path, then follows the goal path."""
    # print(f"goal_path={goal_path}")
    print(f"goal_path.len={len(goal_path)}")
    nearest_idx = find_nearest_path_index(goal_path, current_position)
    # Path from current position to nearest point on goal path
    start_path = [current_position, goal_path[nearest_idx]]
    # Continue along the goal path from the nearest point onward
    print(f"nearest_idx={nearest_idx}")
    # print(f"start_path={start_path}")
    # print(f"goal_path mod ={goal_path[nearest_idx+1:]}")

    if nearest_idx + 1 < len(goal_path):
        final_path = np.vstack((start_path, goal_path[nearest_idx+1:]))
        return final_path
    return np.array(start_path)

def interpolate_path(path, num_points):
    """Interpolates a path using cubic splines and returns the interpolated path."""
    # Separate path coordinates
    x, y, z = path[:, 0], path[:, 1], path[:, 2]
    # Parametrize by cumulative distance
    distances = np.cumsum(np.sqrt(np.diff(x, prepend=x[0])**2 + np.diff(y, prepend=y[0])**2 + np.diff(z, prepend=z[0])**2))
    t = np.linspace(0, distances[-1], num=len(distances))
    interpolator_x = CubicSpline(t, x)
    interpolator_y = CubicSpline(t, y)
    interpolator_z = CubicSpline(t, z)
    t_new = np.linspace(0, distances[-1], num=num_points)
    return np.vstack((interpolator_x(t_new), interpolator_y(t_new), interpolator_z(t_new))).T

class DroneController:
    def __init__(self, client):
        self.client = client
        self.speed = 5.0
        self.tolerance = 0.2
        self.idx = 0

        self.path = []
        self.path_idx = 0


        # If inside container: ip="host.docker.internal", port=41451
        # self._client.confirmConnection()
        # self.client.enableApiControl(True)
        # self.client.takeoffAsync()
        # self.client.armDisarm(True)

    def get_current_position(self):
        """Retrieves the current position of the drone."""
        pos = self.client.getMultirotorState().kinematics_estimated.position
        return np.array([pos.x_val, pos.y_val, pos.z_val])

    def update_path(self, new_path):
        """
        Update the flight path and find the nearest point on the new path.

        Args:
        - new_path (list of list of float): New path as a list of positions.
        """

        # print(f"new_path")
        print(f"new_path={new_path}")
        current_position = self.get_current_position()
        # print(f"init_pos={current_position}")
        joined_path = create_path_to_goal(current_position, new_path)
        # print(f"joined_path={joined_path}")
        # insert one point in between 2 original path points
        interpolated_path = interpolate_path(joined_path, num_points=len(joined_path) * PTS_MULT - 1)
        # print(f"interp_path={interpolated_path}")
        print(f"interp.len={len(interpolated_path)}")
        self.path = interpolated_path
        self.idx = 1

    def calculate_yaw(self, start, end):
        dx, dy = end[0] - start[0], end[1] - start[1]
        yaw = np.degrees(np.arctan2(dy, dx))
        return yaw

    def move_along_path_pos(self):
        
        # print(f"idx={self.idx} path.len={len(self.path)}")
        if self.idx >= len(self.path):
            return False

        current_position = self.get_current_position()
        target_pos = np.array(self.path[self.idx])
        print(f"cur={current_position}")

        # We do not really care about reaching each point. It's may be
        # better to focus on the next point when we are about to reach
        # current one
        if np.linalg.norm(target_pos - current_position) < self.tolerance:
            self.idx += 1
            print(f"Reached index({self.idx})")
            if self.idx >= len(self.path):
                return False

        # current_position = self.get_current_position()
        # print(f"cur_pos={current_position}")
        start = current_position
        end = self.path[self.idx]

        print(f"{start} => {end}")
        
        # Calculate yaw to face the next waypoint
        desired_yaw = self.calculate_yaw(start, end)
        
        # Move to the next position with the calculated yaw and speed
        self.client.moveToPositionAsync(
            end[0], end[1], end[2],
            self.speed,
            yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=desired_yaw)
        ).join()

        current_position = self.get_current_position()
        self.idx += 1
        
        # Brief delay to ensure smooth transition between points
        # time.sleep(0.05)

        return True


class AirSimDatafeedNode:

    def __init__(self):
        rospy.init_node('airsim_data_publisher')

        self.client = airsim.MultirotorClient(ip="host.docker.internal", port=41451)
        self.client.confirmConnection()
        self.ctl = DroneController(self.client)

        self.cam_pcd_pose_pub = rospy.Publisher('/cam_pcd_pose', Pcd2WithPose, queue_size=1)
        self.pointcloud_pub = rospy.Publisher('/point_cloud', PointCloud2, queue_size=1)
        self.path_sub = rospy.Subscriber('/plan_path', Path, self.path_callback)

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

        self.position = [camera_info.pose.position.x_val,
                        camera_info.pose.position.y_val,
                        camera_info.pose.position.z_val]

        self.orientation = [camera_info.pose.orientation.x_val,
                            camera_info.pose.orientation.y_val,
                            camera_info.pose.orientation.z_val,
                            camera_info.pose.orientation.w_val]

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
            stride = 2
            # Compute 3D points from the depth map
            point_cloud = []
            for v in range(0, response.height, stride):
                for u in range(0, response.width, stride):
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

    def path_callback(self, msg):

        positions = []
        # orientations = []
        
        # Iterate through each pose in the Path message
        for pose_stamped in msg.poses:
            # Extract the position and orientation
            pos = pose_stamped.pose.position
            # ori = pose_stamped.pose.orientation
            
            # Append to respective lists
            # positions.append((pos.x, pos.y, pos.z))
            # positions.append((pos.x, pos.y, -pos.z))
            # positions.append((-pos.y, -pos.x, -pos.z))
            # positions.append((-pos.y, pos.x, -pos.z))

            # rot_pos = self.rotate_point(np.array((-pos.y, -pos.x, -pos.z)), axis='x', angle=-90)
            # positions.append((-pos.z, pos.x, pos.y))
            positions.append((pos.y, pos.x, -pos.z))

            # rot_pos = self.rotate_point(np.array((pos.z, pos.y, pos.x)), axis='x', angle=-90)
            # rot_pos = self.rotate_point(np.array((pos.z, pos.y, pos.x)), axis='x', angle=-90)
            # rot_pos = self.rotate_point(np.array((pos.x, -pos.y, -pos.z)), axis='x', angle=-90)
            # positions.append(tuple(rot_pos))
            
            

            # orientations.append((ori.x, ori.y, ori.z, ori.w))
            # orientations.append([0., 0., 0., 1.])

        
        # print(f"plan received:{positions}")

        # if self.sequence % 20 == 0:
        if self.ctl.idx >= len(self.ctl.path):
            print(f"got path len={len(positions)}")
            self.ctl.update_path(positions)

    def run(self):
        rate = rospy.Rate(RATE)

        while not rospy.is_shutdown():
            self.publish_point_cloud_with_pose()

            self.sequence += 1
            # self.ctl.move_along_path()
            self.ctl.move_along_path_pos()
            # self.ctl.step()
            rate.sleep()


if __name__ == '__main__':
    
    # path = [
        # (0., 0, 0),
        # (0., 0, -2),
        # (0., 1, -3),
        # (1., 0, -4),
        # (2., 1, -4),
        # (2., 2, -4),
        # (1., 3, -3),
        # (2., 1, -3),
        # (3., 0, -2),
        # (2., 0, -2),
        # (1., 0, -2),
        # (0., 0, -2)
    # ]

    path = [
        (0., 0, 0),
        (0., 0, -2),
        (0., 0, -3),
        (0., 0, -3),
        (0., 0, -5),
        # (0., 0, -4),
        # (0., 0, -2),
        # (0., 0, -1),
        (0., 0, 0),
    ]

    # orig_path = np.array(path)
    
    # client = airsim.MultirotorClient(ip="host.docker.internal", port=41451)
    # client.confirmConnection()
    # controller = DroneController(client)
    # controller.update_path(orig_path)
    # controller.idx = 0
    # idx = 0
    # while True:
        # if random.randint(0, len(orig_path) * PTS_MULT - 1) == 0:
        # # if idx % 2 == 0:
            # # r_x = random.randint(-5, 5)
            # # r_y = random.randint(-5, 5)
            # # r_z = random.randint(0, 4)
            # # off = np.array([r_x, r_y, r_z])

            # # m_x = random.randint(1, 2)
            # # m_y = random.randint(1, 2)
            # # m_z = random.randint(1, 2)
            # # mul = np.array([m_x, m_y, m_z])

            # # mod_path = (orig_path + off) * mul
            # # controller.update_path(mod_path)

            # controller.update_path(orig_path)


        # controller.move_along_path_pos()
        # # time.sleep(0.05)
        # idx += 1
    
    
    node = AirSimDatafeedNode()
    node.run()
