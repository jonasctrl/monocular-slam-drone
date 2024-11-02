import rospy
from nav_msgs.msg import Path
import airsim
import numpy as np
import time
import math

CAMERA = "fc"

VELOCITY = 0.5
RATE = 1

class AirSimNavigationNode:
    def __init__(self):
        rospy.init_node('airsim_navigation_node')

        self.client = airsim.MultirotorClient(ip="host.docker.internal", port=41451)
        self.client.confirmConnection()

        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()

        self.coordinate_sub = rospy.Subscriber('/plan_map_path', Path, self.coordinate_callback)

        self.is_executing_command = False
        self.current_target_index = 0

        self.target_positions = []
        self.target_orientations = []

        self.pending_target_positions = []
        self.pending_target_orientations = []

        # Movement completion threshold (meters)
        self.position_threshold = 1.0

    def coordinate_callback(self, msg):
        """
        Callback function for receiving path coordinates in ENU format.
        """
        if not self.is_executing_command:
            self.target_positions = [self.t_coordinates(pose.pose.position) for pose in msg.poses]
            self.target_orientations = [self.t_orientation(pose.pose.orientation) for pose in msg.poses]

            rospy.loginfo(f"Received new target path with {len(self.target_positions)} waypoints.")

            self.is_executing_command = True
            self.current_target_index = 0
        else:
            position = self.get_current_position()
            rospy.loginfo(f"Current position: {position}")
            rospy.loginfo(f"Target position: {self.target_positions[self.current_target_index]}")

            self.pending_target_positions = [self.t_coordinates(pose.pose.position) for pose in msg.poses]
            self.pending_target_orientations = [self.t_orientation(pose.pose.orientation) for pose in msg.poses]

            rospy.loginfo("Currently executing a command, storing new path for later.")

    # NOTE: Use this for transformation when debugging
    def t_coordinates(self, point):
        ned_x = point.y
        ned_y = point.x
        ned_z = point.z

        return airsim.Vector3r(ned_x, ned_y, ned_z)

    # NOTE: Use this for transformation when debugging
    def t_orientation(self, orientation):
        return airsim.Quaternionr(orientation.x, orientation.y, orientation.z, orientation.w)

    def quaternion_to_yaw(self, q):
        """
        Convert a quaternion to yaw angle in degrees.
        """
        # AirSim uses Quaternionr(x, y, z, w)
        siny_cosp = 2 * (q.w_val * q.z_val + q.x_val * q.y_val)
        cosy_cosp = 1 - 2 * (q.y_val * q.y_val + q.z_val * q.z_val)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        yaw_deg = math.degrees(yaw)
        return yaw_deg

    def run(self):
        rate = rospy.Rate(RATE)
        while not rospy.is_shutdown():
            if self.is_executing_command and self.target_positions:
                target_position = self.target_positions[self.current_target_index]
                target_orientation = self.target_orientations[self.current_target_index]

                # Yaw angle from the orientation quaternion
                yaw_deg = self.quaternion_to_yaw(target_orientation)
                yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=yaw_deg)

                # Start moving towards the target position with desired yaw
                self.client.moveToPositionAsync(
                    target_position.x_val,
                    target_position.y_val,
                    target_position.z_val,
                    velocity=VELOCITY,
                    yaw_mode=yaw_mode
                )
                rospy.loginfo(f"Started moving towards waypoint {self.current_target_index + 1}/{len(self.target_positions)}: {target_position} with yaw {yaw_deg} degrees")

                # Monitor the movement until completion or collision
                self.monitor_movement(target_position)

                # Move to the next waypoint
                self.current_target_index += 1

                if self.current_target_index >= len(self.target_positions):
                    rospy.loginfo("Completed all waypoints.")

                    # Reset state variables
                    self.is_executing_command = False
                    self.target_positions = []
                    self.target_orientations = []
                    self.current_target_index = 0

                    # Check if there is a pending path
                    if self.pending_target_positions:
                        self.target_positions = self.pending_target_positions
                        self.target_orientations = self.pending_target_orientations
                        self.pending_target_positions = []
                        self.pending_target_orientations = []
                        self.is_executing_command = True
                        rospy.loginfo("Setting pending path as the new target.")
                    else:
                        rospy.loginfo("No pending path.")

            rate.sleep()

    def get_current_position(self):
        # Get the current position of the drone
        state = self.client.getMultirotorState()
        position = state.kinematics_estimated.position
        return position

    def monitor_movement(self, target_position):
        """
        Monitor the drone's movement until it reaches the target or a collision occurs.
        """
        while not rospy.is_shutdown():
            try:
                current_position = self.get_current_position()

                collision_info = self.client.simGetCollisionInfo()
                if collision_info.has_collided:
                    rospy.logwarn("Collision detected!")
                    self.client.cancelLastTask()
                    self.client.hoverAsync().join()
                    break

                distance = np.sqrt(
                    (current_position.x_val - target_position.x_val) ** 2 +
                    (current_position.y_val - target_position.y_val) ** 2 +
                    (current_position.z_val - target_position.z_val) ** 2
                )

                if distance < self.position_threshold:
                    rospy.loginfo("Reached the waypoint.")
                    self.client.hoverAsync().join()
                    break

                time.sleep(0.5)

            except Exception as e:
                rospy.logerr(f"Error in monitor_movement: {str(e)}")

if __name__ == '__main__':
    try:
        node = AirSimNavigationNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
