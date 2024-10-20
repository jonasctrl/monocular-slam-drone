import airsim
from pynput import keyboard
import numpy as np
from scipy.spatial.transform import Rotation as ScipyRotation
import time


class DroneController:
    """
    High level drone controller for manual drone navigation using a regular keyboard.
    """
    def __init__(self):
        self.acceleration = 3.0
        self.max_speed = 6.0
        self.angular_velocity = 90.0
        self.duration = 0.4
        self.friction = 0.5

        self.desired_velocity = np.zeros(3, dtype=np.float32)

        self._key_command_mapping = {
            keyboard.Key.up: "forward",
            keyboard.Key.down: "backward",
            keyboard.Key.left: "turn left",
            keyboard.Key.right: "turn right",
            keyboard.KeyCode.from_char("w"): "up",
            keyboard.KeyCode.from_char("s"): "down",
            keyboard.KeyCode.from_char("a"): "left",
            keyboard.KeyCode.from_char("d"): "right",
        }

        self._active_commands = {command: False for command in self._key_command_mapping.values()}

        # If inside container: ip="host.docker.internal", port=41451
        self._client = airsim.MultirotorClient()
        self._client.confirmConnection()
        self._client.enableApiControl(True)
        self._client.takeoffAsync()

    def fly_by_keyboard(self):
        """
        Begin to listen for keyboard input and send according control commands until `esc` is pressed.
        """
        print("Starting manual control mode...")
        # Start a listener instance that invokes callbacks when keys are pressed or released. When the listener stops,
        # it indicates that the whole execution should stop too.
        with keyboard.Listener(on_press=self._on_press, on_release=self._on_release) as keyboard_listener:
            keyboard_listener.wait()
            print("Ready, you can control the drone by keyboard now.")
            while keyboard_listener.running:
                self._handle_commands()
                time.sleep(self.duration / 2.0)
            keyboard_listener.join()
        print("Manual control mode was successfully deactivated.")

    def move(self, velocity, yaw_rate):
        self._client.moveByVelocityAsync(velocity[0].item(), velocity[1].item(), velocity[2].item(), self.duration,
                                         drivetrain=airsim.DrivetrainType.ForwardOnly, 
                                         yaw_mode=airsim.YawMode(True, yaw_rate))

    def _on_press(self, key):
        if key in self._key_command_mapping.keys():
            self._active_commands[self._key_command_mapping[key]] = True
        elif key == keyboard.Key.esc:
            # Shutdown.
            return False

    def _on_release(self, key):
        if key in self._key_command_mapping.keys():
            self._active_commands[self._key_command_mapping[key]] = False

    def _handle_commands(self):
        drone_orientation = ScipyRotation.from_quat(self._client.simGetVehiclePose().orientation.to_numpy_array())
        yaw = drone_orientation.as_euler('zyx')[0]
        forward_direction = np.array([np.cos(yaw), np.sin(yaw), 0])
        left_direction = np.array([np.cos(yaw - np.deg2rad(90)), np.sin(yaw - np.deg2rad(90)), 0])

        if self._active_commands["forward"] or self._active_commands["backward"]:
            forward_increment = forward_direction * self.duration * self.acceleration
            if self._active_commands["forward"]:
                self.desired_velocity += forward_increment
            else:
                self.desired_velocity -= forward_increment
        else:
            forward_component = np.dot(self.desired_velocity, forward_direction) * forward_direction
            self.desired_velocity -= self.friction * forward_component

        if self._active_commands["up"] or self._active_commands["down"]:
            vertical_component = drone_orientation.apply(np.array([0.0, 0.0, -1.0]))
            vertical_component *= self.duration * self.acceleration
            if self._active_commands["up"]:
                self.desired_velocity += vertical_component
            else:
                self.desired_velocity -= vertical_component
        else:
            self.desired_velocity[2] *= self.friction

        if self._active_commands["left"] or self._active_commands["right"]:
            lateral_increment = left_direction * self.duration * self.acceleration
            if self._active_commands["left"]:
                self.desired_velocity += lateral_increment
            else:
                self.desired_velocity -= lateral_increment
        else:
            left_component = np.dot(self.desired_velocity, left_direction) * left_direction
            self.desired_velocity -= self.friction * left_component

        speed = np.linalg.norm(self.desired_velocity)
        if speed > self.max_speed:
            self.desired_velocity = self.desired_velocity / speed * self.max_speed

        yaw_rate = 0.0
        if self._active_commands["turn left"]:
            yaw_rate = -self.angular_velocity
        elif self._active_commands["turn right"]:
            yaw_rate = self.angular_velocity

        self.move(self.desired_velocity, yaw_rate)


if __name__ == "__main__":
    controller = DroneController()
    controller.fly_by_keyboard()