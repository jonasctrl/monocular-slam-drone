import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image, Imu
import cv2
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Vector3, Point
import os
import airsim
import numpy as np
import time
import math

IMAGE_SAVE_FOLDER = '/opt/scripts/images/'
CAMERA = "front-center"
RATE = 200  

class AirSimDatafeedNode:

    def __init__(self):
        rospy.init_node('airsim_data_publisher')

        self.client = airsim.MultirotorClient(ip="host.docker.internal", port=41451)
        self.client.confirmConnection()

        self.image_pub = rospy.Publisher('/cam0/image_raw', Image, queue_size=1)
        self.depthimage_pub = rospy.Publisher('/cam0/image_depth_raw', Image, queue_size=1)
        self.imu_pub = rospy.Publisher('/imu0', Imu, queue_size=1)
        self.rpy_pub = rospy.Publisher('/rpy', Vector3, queue_size=10)
        self.position_pub = rospy.Publisher('/position', Point, queue_size=10)

        self.bridge = CvBridge()
        self.image_type = airsim.ImageType.Scene

        self.frame_count = 0
        self.sequence = 0
        self.start_time = time.time()

    def publish_image(self, response, type_msg, img_rgb, imgEncoding, method):
        image_filename = os.path.join(IMAGE_SAVE_FOLDER, type_msg + '_' + str(time.time()) +'.png')
        cv2.imwrite(image_filename, img_rgb)
        
        img_data = self.bridge.cv2_to_imgmsg(img_rgb, encoding=imgEncoding)
        img_data.header.seq = self.sequence
        img_data.header.frame_id = "camera"
        img_data.header.stamp = rospy.Time.now()

        method.publish(img_data)
        print("Image Published"
            )
        rospy.loginfo("Image saved successfully to %s", image_filename)

    def publish_rgb(self):
        try:
            response = self.client.simGetImages([airsim.ImageRequest(0, self.image_type, pixels_as_float=False, compress=False)])[0]
            img_rgb = np.frombuffer(response.image_data_uint8, dtype=np.uint8).reshape(response.height, response.width, 3)
            self.publish_image(response, 'rgb', img_rgb, 'bgr8', self.image_pub)
        except CvBridgeError as e:
            rospy.logerr("CvBridge error: %s", e)

    def publish_rgbd(self):
        try:
            response = self.client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.DepthPlanar, pixels_as_float=True, compress=False)])[0]
            img_rgb = np.array(response.image_data_float, dtype=np.float32).reshape(response.height, response.width)
            self.publish_image(response, 'rgbd', img_rgb, '32FC1', self.depthimage_pub)
        except CvBridgeError as e:
            rospy.logerr("CvBridge error: %s", e)
            
    def publish_imu(self):
        imu_data = self.client.getImuData()
    
        imu_msg = Imu()
        imu_msg.header.seq = self.sequence
        imu_msg.header.stamp = rospy.Time.now()
        imu_msg.header.frame_id = "camera"
        
        imu_msg.linear_acceleration.x = imu_data.linear_acceleration.x_val
        imu_msg.linear_acceleration.y = imu_data.linear_acceleration.y_val
        imu_msg.linear_acceleration.z = imu_data.linear_acceleration.z_val
        
        imu_msg.angular_velocity.x = imu_data.angular_velocity.x_val
        imu_msg.angular_velocity.y = imu_data.angular_velocity.y_val
        imu_msg.angular_velocity.z = imu_data.angular_velocity.z_val
        
        imu_msg.orientation.w = imu_data.orientation.w_val
        imu_msg.orientation.x = imu_data.orientation.x_val
        imu_msg.orientation.y = imu_data.orientation.y_val
        imu_msg.orientation.z = imu_data.orientation.z_val
        
        imu_msg.orientation_covariance = [-1] * 9
        imu_msg.angular_velocity_covariance = [-1] * 9
        imu_msg.linear_acceleration_covariance = [-1] * 9

        self.imu_pub.publish(imu_msg)

    def publish_state(self):
        state = self.client.getMultirotorState()

        orientation = state.kinematics_estimated.orientation

        x = orientation.x_val
        y = orientation.y_val
        z = orientation.z_val
        w = orientation.w_val

        # Roll 
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(t0, t1)

        # Pitch 
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = math.asin(t2)

        # Yaw 
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)

        roll_deg = math.degrees(roll)
        pitch_deg = math.degrees(pitch)
        yaw_deg = math.degrees(yaw)

        rpy_msg = Vector3()
        rpy_msg.x = roll_deg
        rpy_msg.y = pitch_deg
        rpy_msg.z = yaw_deg

        self.rpy_pub.publish(rpy_msg);

    def publish_position(self):
        camera_info = self.client.simGetCameraInfo(CAMERA, "")
        camera_position = camera_info.pose.position

        point_msg = Point()
        point_msg.x = camera_position.x_val
        point_msg.y = camera_position.y_val
        point_msg.z = camera_position.z_val

        self.position_pub.publish(point_msg)


    def run(self):
        rate = rospy.Rate(RATE)
        image_counter = 0
        
        while not rospy.is_shutdown():
            self.publish_imu()
            self.publish_state()
            self.publish_position()
            
            if image_counter == 0:
                self.publish_rgb()
                self.publish_rgbd()

            self.frame_count += 1
            self.sequence += 1

            elapsed_time = time.time() - self.start_time
            current_fps = self.frame_count / elapsed_time

            if self.sequence % RATE == 0:
                print(
                    f"Iteration {self.sequence}:"
                    f" Average FPS: {current_fps:.2f}"
                )

            rate.sleep()

if __name__ == '__main__':
    node = AirSimDatafeedNode()
    node.run()