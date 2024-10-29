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

    def ned_to_enu_imu(self, imu_data):
        """
        Convert IMU data from NED to ENU coordinate system.
        """
        # Swap axes and negate Z values for both linear acceleration and angular velocity
        linear_acceleration_enu = airsim.Vector3r(imu_data.linear_acceleration.y_val,
                                                  imu_data.linear_acceleration.x_val,
                                                  -imu_data.linear_acceleration.z_val)

        angular_velocity_enu = airsim.Vector3r(imu_data.angular_velocity.y_val,
                                               imu_data.angular_velocity.x_val,
                                               -imu_data.angular_velocity.z_val)

        # Swap the quaternion's x and y, and negate z for orientation conversion
        orientation_enu = airsim.Quaternionr(imu_data.orientation.y_val,
                                             imu_data.orientation.x_val,
                                             -imu_data.orientation.z_val,
                                             imu_data.orientation.w_val)

        return linear_acceleration_enu, angular_velocity_enu, orientation_enu

    def publish_IMU(self):
        imu_data = self.client.getImuData()

        # Convert IMU data from NED to ENU
        linear_acceleration_enu, angular_velocity_enu, orientation_enu = self.ned_to_enu_imu(imu_data)

        imu_msg = Imu()
        imu_msg.header.seq = self.sequence
        imu_msg.header.stamp = rospy.Time.now()
        imu_msg.header.frame_id = "camera"

        # Set linear acceleration in ENU
        imu_msg.linear_acceleration.x = linear_acceleration_enu.x_val
        imu_msg.linear_acceleration.y = linear_acceleration_enu.y_val
        imu_msg.linear_acceleration.z = linear_acceleration_enu.z_val

        # Set angular velocity in ENU
        imu_msg.angular_velocity.x = angular_velocity_enu.x_val
        imu_msg.angular_velocity.y = angular_velocity_enu.y_val
        imu_msg.angular_velocity.z = angular_velocity_enu.z_val

        # Set orientation in ENU
        imu_msg.orientation.w = orientation_enu.w_val
        imu_msg.orientation.x = orientation_enu.x_val
        imu_msg.orientation.y = orientation_enu.y_val
        imu_msg.orientation.z = orientation_enu.z_val

        # Set covariance (defaulting to -1 since it's not provided by AirSim)
        imu_msg.orientation_covariance = [-1] * 9
        imu_msg.angular_velocity_covariance = [-1] * 9
        imu_msg.linear_acceleration_covariance = [-1] * 9

        # Publish the IMU message
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
