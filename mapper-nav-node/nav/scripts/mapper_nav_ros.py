#!/usr/bin/env python3

import math
import struct
import time
import numpy as np

import rospy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2 as pc2
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PointStamped, Point
from std_msgs.msg import Header


# from cv_bridge import CvBridge
import airsim

from mapper import VoxArray, precompile
from modules.depth_anything_module import DepthAnythingEstimatorModule
import nav_config as cfg


HEIGHT, WIDTH = 144, 256
CAMERA = "front-center"
RATE = 5
PTS_MULT = 2

g_num = 0


class MapperNavNode:
    def __init__(self):
        self.sequence = 0
        self.vmap = VoxArray(resolution=cfg.map_resolution, shape=(cfg.map_width,cfg.map_depth,cfg.map_heigth))
        rospy.init_node('mapper_nav', anonymous=True)

        self.depth_estimator_module = DepthAnythingEstimatorModule()
        
        self.client = airsim.MultirotorClient(ip="host.docker.internal", port=41451)
        self.client.confirmConnection()
        self.ctl = DroneController(self.client)

        # self.bridge = CvBridge()
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

    ################
    #  PUBLISHERS  #
    ################

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
        points, intensities = self.vmap.get_occupied_space_pcd()
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "map"

        if cfg.publish_occup_intensities:
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
            ]

            # Pack data into binary format
            data = []
            for point, intensity in zip(points, intensities):
                data.append(struct.pack('ffff', point[0], point[1], point[2], intensity))

            data = b''.join(data)

            pcd_msg = PointCloud2(
                header=header,
                height=1,
                width=len(points),
                fields=fields,
                is_bigendian=False,
                point_step=16,  # 4 fields * 4 bytes/field
                row_step=16 * len(points),
                data=data,
                is_dense=True
            )
        else:
            pcd_msg = pc2.create_cloud_xyz32(header, points)

        self.occupied_pub.publish(pcd_msg)

    def publish_empty_space_msg(self):
        points = self.vmap.get_empty_space_pcd()
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "map"
        pcd_msg = pc2.create_cloud_xyz32(header, points)

        self.empty_pub.publish(pcd_msg)

    ################
    #  SUBSCRIBERS #
    ################

    def goal_callback(self, msg):
        position = msg.pose.position
        # pos = tuple(np.array([position.x, position.y, position.z]).astype(int))
        pos = tuple(np.array([position.x, position.y, self.vmap.cntr[2]]).astype(int))
        self.vmap.set_goal(pos, update_start=True)

    ######################
    #  DEPTH ESTIMATION  #
    ######################
    
    def estimate_depth(self):
        response = self.client.simGetImages([
            airsim.ImageRequest(CAMERA, airsim.ImageType.Scene, pixels_as_float=False, compress=False)
            ])[0]

        # if response.width == 0:
            # rospy.logwarn("Failed to get valid image from AirSim camera")
            # return None

        img_rgb = np.frombuffer(response.image_data_uint8, dtype=np.uint8).reshape(response.height, response.width, 3)

        depth_map = self.depth_estimator_module.generate_depth_map(img_rgb)

        return depth_map
        
    
    def get_direct_depth(self):
        response = self.client.simGetImages(
            [airsim.ImageRequest(CAMERA, airsim.ImageType.DepthPlanar, pixels_as_float=True, compress=False)])[0]

        # if response.width == 0:
            # rospy.logwarn("Failed to get valid image from AirSim camera")
            # return None

        depth_map= np.array(response.image_data_float, dtype=np.float32).reshape(response.height, response.width)
        return depth_map
        

    def depth_img_to_pcd(self, dimg, fov):
        (image_height, image_width) = dimg.shape

        # Compute fx and fy from FOV
        hfov_rad = fov * math.pi / 180.0
        fx = (image_width / 2.0) / math.tan(hfov_rad / 2.0)
        fy = fx
        # Compute principal point coordinates
        cx = image_width / 2.0
        cy = image_height / 2.0

        # Compute 3D points from the depth map
        pcd = []
        for v in range(0, image_height, cfg.dimg_stride):
            for u in range(0, image_width, cfg.dimg_stride):
                z = dimg[v, u]
                if z >= cfg.dimg_min_depth and z < cfg.dimg_max_depth:
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy
                    # rotation 90 deg OX
                    pcd.append([z, -x, -y])

        return pcd
        

    def get_camera_info(self):
        camera_info = self.client.simGetCameraInfo(CAMERA)
        fov = camera_info.fov

        p = camera_info.pose.position
        # NED to ENU
        pt = np.array([p.x_val, -p.y_val, -p.z_val])
        o = camera_info.pose.orientation
        # NED to ENU
        ori = np.array([o.x_val, -o.y_val, -o.z_val, o.w_val])

        return fov, pt, ori


    def step(self):
        t0 = time.time()
        fov, position, orientation = self.get_camera_info()

        if cfg.use_rgb_imaging:
            depth_data = self.estimate_depth()
        else:
            depth_data = self.get_direct_depth()

        if depth_data is None:
            rospy.logwarn("No depth data")
        

        t1 = time.time()

        pcd = self.depth_img_to_pcd(depth_data, fov)
        
        pos = position
        qtr = orientation

        t2 = time.time()

        ch_pts = self.vmap.update(pcd, pos, qtr, False)
        
        t3 = time.time()
        self.vmap.plan(ch_pts)
        t4 = time.time()


        path, _ = self.vmap.get_plan(orig_coord=True)

        if len(path) > 1:
            self.ctl.path = path
            self.ctl.move_along_path_pos()


        t5 = time.time()
        
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

        t6 = time.time()

        print(f"img: {round(t1-t0, 3)} to_pcd:{round(t2-t1, 3)} map:{round(t3-t2, 3)} plan:{round(t4-t3, 3)} move:{round(t5-t4, 3)} pub:{round(t6-t5, 3)}")
            


    def run(self):
        rate = rospy.Rate(RATE)

        while not rospy.is_shutdown():
            self.step()

            self.sequence += 1
            rate.sleep()


##################
#  DRONE CONTROL #
##################

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
        path = [airsim.Vector3r(x, -y, -z) for (x, y, z) in self.path]
        # print(f"air_path={path}")
        self.client.moveOnPathAsync(
            path=path,
            velocity=self.speed,
            timeout_sec=10,
            drivetrain=airsim.DrivetrainType.ForwardOnly,
            yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0),
            lookahead=-1,
            adaptive_lookahead=0
        )


if __name__ == '__main__':
    try:
        print(f"using rgb: {cfg.use_rgb_imaging}")
        precompile()

        node = MapperNavNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
