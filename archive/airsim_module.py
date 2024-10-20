import airsim
import cv2
import numpy as np
import time

default_camera_name = "front-center"


class AirSimModule:
    def __init__(self, camera_name=default_camera_name):
        self.client = airsim.MultirotorClient()
        self.camera_name = camera_name

    def initialize(self):
        self.client.confirmConnection()
        print("Connected!")

    def get_camera_image(self):
        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest(self.camera_name, airsim.ImageType.Scene, False, False)
            ])

            response = responses[0]
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img_rgb = img1d.reshape(response.height, response.width, 3)

            return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Error capturing image: {str(e)}")
            return None

    def get_ground_truth_depth_image(self):
        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest(self.camera_name, airsim.ImageType.DepthPerspective, True, False)
            ])

            response = responses[0]
            img1d = np.array(response.image_data_float, dtype=np.float32)
            img_rgb = img1d.reshape(response.height, response.width)

            return img_rgb
        except Exception as e:
            print(f"Error capturing ground truth depth image: {str(e)}")

    def get_telemetry(self):
        try:
            camera_info = self.client.simGetCameraInfo(self.camera_name)
            telemetry = {"timestamp": time.time()}

            if camera_info is not None and hasattr(camera_info, 'pose'):
                camera_pose = camera_info.pose
                camera_position = camera_pose.position
                camera_orientation = camera_pose.orientation

                telemetry["camera_position"] = {
                    "x": camera_position.x_val,
                    "y": camera_position.y_val,
                    "z": camera_position.z_val
                }
                telemetry["camera_orientation"] = {
                    "x": camera_orientation.x_val,
                    "y": camera_orientation.y_val,
                    "z": camera_orientation.z_val,
                    "w": camera_orientation.w_val
                }

            if hasattr(camera_info, 'fov'):
                telemetry["camera_fov"] = camera_info.fov

            return telemetry
        except Exception as e:
            print(f"Error getting telemetry: {str(e)}")
            return None
