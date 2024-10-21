import rospy
from cv_bridge import CvBridge
import airsim
from sensor_msgs.msg import Image, Imu

CAMERA = "front-center"
RATE = 200

class AirSimDatafeedNode:

    def __init__(self):
        rospy.init_node('airsim_camera_publisher')

        self.client = airsim.MultirotorClient(ip="host.docker.internal", port=41451)
        self.client.confirmConnection()

        self.imu_pub = rospy.Publisher('/imu0', Imu, queue_size=1)

        self.bridge = CvBridge()
        self.sequence = 0

    def publish_IMU(self):
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
        
    def run(self):
        rate = rospy.Rate(RATE)

        while not rospy.is_shutdown():
            self.publish_IMU()

            self.sequence += 1
            rate.sleep()

if __name__ == '__main__':
    node = AirSimDatafeedNode()
    node.run()
