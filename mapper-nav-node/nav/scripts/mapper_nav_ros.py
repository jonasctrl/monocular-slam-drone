#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2 as pc2
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PointStamped, Point
from std_msgs.msg import Header
# from nav.msg import Pcd2WithPose, DepthWithPose
from cv_bridge import CvBridge
import math

import airsim
# import math
import numpy as np
# from scipy.spatial.transform import Rotation as R
import time

from mapper import VoxArray 

import nav_config as cfg


HEIGHT, WIDTH = 144, 256
# CAMERA = "fc"
CAMERA = "front-center"
RATE = 5
PTS_MULT = 2

g_num = 0


class MapperNavNode:
    def __init__(self):
        self.sequence = 0
        self.vmap = VoxArray(resolution=cfg.map_resolution, shape=[600,600,300])
        rospy.init_node('mapper_nav', anonymous=True)
        
        self.client = airsim.MultirotorClient(ip="host.docker.internal", port=41451)
        self.client.confirmConnection()
        self.ctl = DroneController(self.client)

        self.bridge = CvBridge()
        self.occupied_pub = rospy.Publisher('/occupied_space', PointCloud2, queue_size=1)
        self.empty_pub = rospy.Publisher('/empty_space', PointCloud2, queue_size=1)
        self.cam_path_pub = rospy.Publisher('/cam_path', Path, queue_size=1)
        self.plan_path_pub = rospy.Publisher('/plan_path', Path, queue_size=1)
        self.plan_map_path_pub = rospy.Publisher('/plan_map_path', Path, queue_size=1)
        self.pose_pub = rospy.Publisher('/map_pose', PoseStamped, queue_size=1)

        self.start_pub = rospy.Publisher('/nav_start', PointStamped, queue_size=1)
        self.goal_pub = rospy.Publisher('/nav_goal', PointStamped, queue_size=1)

        # self.depth_sub = rospy.Subscriber('/ground_truth/depth_with_pose', DepthWithPose, self.image_callback)
        # self.depth_sub = rospy.Subscriber('/cam_pcd_pose', Pcd2WithPose, self.pcd_pose_callback, queue_size=1)
        self.goal_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback, queue_size=1)
        # self.depth_sub = rospy.Subscriber('/cam_pcd_pose', Pcd2WithPose, self.pcd_pose_callback, queue_size=1)

        rospy.loginfo("Mapper-Navigation node initialized.")


    def get_camera_info(self):
        camera_info = self.client.simGetCameraInfo(CAMERA)
        fov = camera_info.fov

        p = camera_info.pose.position
        # NED to ENU
        pt = np.array([-p.x_val, p.y_val, -p.z_val])
        o = camera_info.pose.orientation
        # NED to ENU
        ori = np.array([-o.x_val, o.y_val, -o.z_val, o.w_val])

        return fov, pt, ori


    def step(self):
        if True:
        # try:
            # print(f"step[{self.sequence}]")
            # Get camera position and orientation
            fov, position, orientation = self.get_camera_info()

            response = self.client.simGetImages(
                [airsim.ImageRequest(CAMERA, airsim.ImageType.DepthPlanar, pixels_as_float=True, compress=False)])[0]
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

            # Compute 3D points from the depth map
            pcd = []
            for v in range(0, response.height, cfg.dimg_stride):
                for u in range(0, response.width, cfg.dimg_stride):
                    # print(f"{[v, u, depth_data[v, u]]}")
                    z = depth_data[v, u]
                    if z >= cfg.dimg_min_depth and z < cfg.dimg_max_depth:
                        x = (u - cx) * z / fx
                        y = (v - cy) * z / fy
                        # rotation 90 deg OX
                        pcd.append([-z, x, -y])
            
            pos = position
            qtr = orientation
            is_glob_fame = False

            t1 = time.time()

    
            # print(f"step[{self.sequence}] updating map")
            ch_pts = self.vmap.update(pcd, pos, qtr, is_glob_fame)
            
            t2 = time.time()
            # print(f"step[{self.sequence}] planning")
            self.vmap.plan(ch_pts)
            t3 = time.time()


            # print(f"step[{self.sequence}] getting plan")
            path, _ = self.vmap.get_plan(orig_coord=True)
            # print(f"path={path}")

            # print(f"step[{self.sequence}] moving")
            if len(path) > 1:
                self.ctl.path = path
                self.ctl.move_along_path_pos()

                # self.vmap.plan_path = []
                # self.vmap.plan_qtrs = []



            t4 = time.time()

            
            # print(f"step[{self.sequence}] publishing")
            if cfg.publish_occup:
                self.publish_occupied_space_msg()

            if cfg.publish_empty:
                self.publish_empty_space_msg()
            
            if cfg.publish_pose:
                self.publish_map_pose_msg()


            if cfg.publish_path:
                self.publish_cam_path_msg()

            if cfg.publish_plan:
                self.publish_start_msg()
                self.publish_goal_msg()
                self.publish_plan_path_msg(orig=False) # Map scale
                self.publish_plan_path_msg(orig=True) # Original scale

            t5 = time.time()

            print(f"mapping:{round(t2-t1, 4)} planning:{round(t3-t2, 4)} move:{round(t4-t3, 4)} pub:{round(t5-t4, 4)}")
            

        # except Exception as e:
            # rospy.logerr(f"Error in step: {str(e)}")


    def publish_cam_path_msg(self):
        path_msg = Path()

        path_msg.header = Header()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = rospy.Time.now()

        # print(f"cam_poses={self.vmap.cam_poses}")
        for pt, qtr in zip(self.vmap.cam_path_acc, self.vmap.cam_qtrs):
            (tx, ty, tz) = pt
            (qx, qy, qz, qw) = qtr

            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = "map"
            pose_stamped.header.stamp = rospy.Time.now()

            pose_stamped.pose.position.x = tx
            pose_stamped.pose.position.y = ty
            pose_stamped.pose.position.z = tz

            pose_stamped.pose.orientation.x = qx
            pose_stamped.pose.orientation.y = qy
            pose_stamped.pose.orientation.z = qz
            pose_stamped.pose.orientation.w = qw

            path_msg.poses.append(pose_stamped)

        self.cam_path_pub.publish(path_msg)

    def publish_plan_path_msg(self, orig=True):
        path_msg = Path()

        path_msg.header = Header()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = rospy.Time.now()

        path, qtrs = self.vmap.get_plan(orig_coord=orig)
        for pt, qtr in zip(path, qtrs):
            (tx, ty, tz) = pt
            (qx, qy, qz, qw) = qtr

            # Need to rotate 90 deg in Z axis
            # orig_qtr = R.from_quat(qtr)
            # z_rot = R.from_euler('z', np.deg2rad(90))
            # rot_qtr = z_rot * orig_qtr
            # (qx, qy, qz, qw) = rot_qtr.as_quat()
            
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = "map"
            pose_stamped.header.stamp = rospy.Time.now()

            pose_stamped.pose.position.x = tx
            pose_stamped.pose.position.y = ty
            pose_stamped.pose.position.z = tz

            pose_stamped.pose.orientation.x = qx
            pose_stamped.pose.orientation.y = qy
            pose_stamped.pose.orientation.z = qz
            pose_stamped.pose.orientation.w = qw

            path_msg.poses.append(pose_stamped)

        if orig:
            self.plan_path_pub.publish(path_msg)
        else:
            # print(f"path_msg={path_msg}")
            self.plan_map_path_pub.publish(path_msg)

    def publish_map_pose_msg(self):
        (pt, qtr) = self.vmap.get_pose(orig_coord=False)
        (tx, ty, tz) = pt
        (qx, qy, qz, qw) = qtr

        # print(f"cur\t qtr={qtr}")
        # print(f"lpt\t qtr={self.vmap.cam_qtrs[-1]}")

        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "map"
        pose_stamped.header.stamp = rospy.Time.now()

        pose_stamped.pose.position.x = tx
        pose_stamped.pose.position.y = ty
        pose_stamped.pose.position.z = tz

        pose_stamped.pose.orientation.x = qx
        pose_stamped.pose.orientation.y = qy
        pose_stamped.pose.orientation.z = qz
        pose_stamped.pose.orientation.w = qw

        self.pose_pub.publish(pose_stamped)


    def publish_start_msg(self):
        (x, y, z) = self.vmap.get_start(orig_coord=False)
        point_msg = PointStamped()
        point_msg.header.stamp = rospy.Time.now()
        point_msg.header.frame_id = "map"  # Adjust to your frame of reference
        point_msg.point = Point(x, y, z)
        self.start_pub.publish(point_msg)

    def publish_goal_msg(self):
        (x, y, z) = self.vmap.get_goal(orig_coord=False)
        point_msg = PointStamped()
        point_msg.header.stamp = rospy.Time.now()
        point_msg.header.frame_id = "map"  # Adjust to your frame of reference
        point_msg.point = Point(x, y, z)
        self.goal_pub.publish(point_msg)

    def publish_occupied_space_msg(self):
        points = self.vmap.get_occupied_space_pcd()
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "map"
        pcd_msg = pc2.create_cloud_xyz32(header, points)

        self.occupied_pub.publish(pcd_msg)

    def publish_empty_space_msg(self):
        points = self.vmap.get_empty_space_pcd()
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "map"
        pcd_msg = pc2.create_cloud_xyz32(header, points)

        self.empty_pub.publish(pcd_msg)

    def goal_callback(self, msg):
        position = msg.pose.position
        # pos = tuple(np.array([position.x, position.y, position.z]).astype(int))
        pos = tuple(np.array([position.x, position.y, 77]).astype(int))
        self.vmap.set_goal(pos, update_start=True)


    def run(self):
        rate = rospy.Rate(RATE)

        while not rospy.is_shutdown():
            self.step()

            self.sequence += 1
            rate.sleep()


##############
#  pathhing  #
##############

class DroneController:
    def __init__(self, client):
        self.client = client
        self.speed = 5.0
        self.path = []


        # If inside container: ip="host.docker.internal", port=41451
        # self._client.confirmConnection()
        # self.client.enableApiControl(True)
        # self.client.takeoffAsync()
        # self.client.armDisarm(True)

    def move_along_path_pos(self):
        path = [airsim.Vector3r(-x, y, -z) for (x, y, z) in self.path]
        # print(f"air_path={path}")
        self.client.moveOnPathAsync(
            path=path,
            velocity=self.speed,
            timeout_sec=60,
            drivetrain=airsim.DrivetrainType.ForwardOnly,
            yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0),
            lookahead=-1,
            adaptive_lookahead=0
        )


if __name__ == '__main__':
    try:
        node = MapperNavNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
