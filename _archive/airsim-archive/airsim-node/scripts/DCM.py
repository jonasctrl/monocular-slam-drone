#!/usr/bin/env python3
import airsim
import rospy
from geometry_msgs.msg import PoseStamped
import numpy as np
import threading
import queue

RATE = 10 
STEP_SIZE = 0.5 
TOLERANCE = 0.05 

class AirSimNavigator:
    def __init__(self):
        self.airsim_thread = threading.Thread(target=self.airsim_loop)
        self.direction_queue = queue.Queue()
        self.current_goal = None
        self.should_exit = False

        rospy.init_node('airsim_navigator', anonymous=True)
        rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_callback)

    def goal_callback(self, msg):
        goal = np.array([msg.pose.position.x, msg.pose.position.y, -msg.pose.position.z])
        # Only update if the new goal is different from the current goal
        if self.current_goal is None or not np.allclose(self.current_goal, goal, atol=TOLERANCE):
            self.current_goal = goal
            rospy.loginfo(f"Received new goal: {goal}")

    def airsim_loop(self):
        client = airsim.MultirotorClient(ip="host.docker.internal", port=41451)
        client.confirmConnection()
        client.enableApiControl(True)
        client.armDisarm(True)

        rospy.loginfo("Taking off...")
        client.takeoffAsync().join()
        rospy.loginfo("Takeoff complete")

        while not self.should_exit:
            if self.current_goal is not None:
                self.navigate_to_goal(client)

            rospy.sleep(1.0 / RATE)

    def get_current_position(self, client):
        drone_state = client.getMultirotorState()
        return np.array([
            drone_state.kinematics_estimated.position.x_val,
            drone_state.kinematics_estimated.position.y_val,
            drone_state.kinematics_estimated.position.z_val
        ])

    def navigate_to_goal(self, client):
        current_position = self.get_current_position(client)
        direction = self.current_goal - current_position
        distance_to_goal = np.linalg.norm(direction)

        if distance_to_goal <= TOLERANCE:
            rospy.loginfo("Goal reached")
            self.current_goal = None
            return

        move_distance = min(STEP_SIZE, distance_to_goal)
        direction_normalized = direction / distance_to_goal
        target_position = current_position + direction_normalized * move_distance

        rospy.loginfo(f"Current position: {current_position}")
        rospy.loginfo(f"Moving towards: {target_position} (Goal: {self.current_goal})")
        
        client.moveToPositionAsync(
            target_position[0], target_position[1], target_position[2],
            0.1,  # velocity
        ).join()

    def run(self):
        self.airsim_thread.start()
        rospy.spin()
        self.should_exit = True
        self.airsim_thread.join()

if __name__ == "__main__":
    try:
        navigator = AirSimNavigator()
        navigator.run()
    except rospy.ROSInterruptException:
        pass
