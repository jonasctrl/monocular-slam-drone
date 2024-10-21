import airsim
import numpy as np
import time

class DepthCalibrator:
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.camera_name = "front-center"
        self.pixels_per_meter = None

    def initialize_drone(self):
        print("Initializing drone...")
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        print("Drone initialized and ready.")

    def get_depth_image(self):
        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest(self.camera_name, airsim.ImageType.DepthPerspective, True)
            ])
            depth_image = airsim.list_to_2d_float_array(responses[0].image_data_float, responses[0].width,
                                                        responses[0].height)
            return depth_image
        except Exception as e:
            print(f"Error capturing depth image: {str(e)}")
            return None

    def calculate_pixels_per_meter(self):
        print("Calculating pixels per meter...")

        # Move to initial position
        self.client.moveToPositionAsync(0, 0, -5, 1).join()
        time.sleep(2)
        initial_depth = self.get_depth_image()

        if initial_depth is None:
            print("Failed to get initial depth image")
            return

        # Move forward by 1 meter
        self.client.moveToPositionAsync(1, 0, -5, 1).join()
        time.sleep(2)
        final_depth = self.get_depth_image()

        if final_depth is None:
            print("Failed to get final depth image")
            return

        # Calculate the difference
        depth_diff = final_depth - initial_depth

        # Count the number of pixels that changed significantly
        changed_pixels = np.sum(np.abs(depth_diff) > 0.1)  # Threshold of 0.1 for change detection

        # Calculate pixels per meter
        self.pixels_per_meter = changed_pixels

        print(f"Pixels per meter: {self.pixels_per_meter}")

    def run_calibration(self):
        self.initialize_drone()
        self.calculate_pixels_per_meter()

        if self.pixels_per_meter is not None:
            print(f"Calibration complete. Pixels per meter: {self.pixels_per_meter}")
        else:
            print("Calibration failed.")

        # Land the drone
        self.client.landAsync().join()
        self.client.armDisarm(False)
        self.client.enableApiControl(False)


def main():
    calibrator = DepthCalibrator()
    calibrator.run_calibration()


if __name__ == "__main__":
    main()