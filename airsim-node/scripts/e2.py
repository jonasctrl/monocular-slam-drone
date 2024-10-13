#!/usr/bin/env python3
import airsim
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import math

def calculate_focal_length(fov_degrees, image_width):
    # Calculate focal length in pixels
    fov_rad = math.radians(fov_degrees)
    focal_length = (image_width / 2) / math.tan(fov_rad / 2)
    return focal_length

def main():
    rospy.init_node('airsim_to_orbslam3_feed', anonymous=True)
    image_pub = rospy.Publisher('/cam0/image_raw', Image, queue_size=1)
    bridge = CvBridge()

    client = airsim.MultirotorClient(ip="host.docker.internal", port=41451)
    client.confirmConnection()

    camera_name = "0"
    image_type = airsim.ImageType.Scene

    # Retrieve camera info from AirSim
    camera_info = client.simGetCameraInfo(camera_name)
    fov_degrees = camera_info.fov
    rospy.loginfo(f"Camera FOV: {fov_degrees} degrees")

    # Retrieve image size from a sample image
    image_request = airsim.ImageRequest(camera_name, image_type, pixels_as_float=False, compress=False)
    responses = client.simGetImages([image_request])
    if responses and len(responses) > 0 and responses[0].width != 0:
        response = responses[0]
        image_width = response.width
        image_height = response.height
        rospy.loginfo(f"Camera dimensions: {image_width}x{image_height}")
    else:
        # Set default dimensions if not available
        image_width, image_height = 640, 480  # Default values
        rospy.logwarn(f"Camera dimensions not available. Using default {image_width}x{image_height}")

    # Calculate focal length
    focal_length = calculate_focal_length(fov_degrees, image_width)
    rospy.loginfo(f"Calculated focal length: {focal_length} pixels")

    # Print camera parameters for ORB-SLAM3 YAML file
    rospy.loginfo("Camera parameters to use in ORB-SLAM3 YAML file:")
    rospy.loginfo(f"Camera.fx: {focal_length}")
    rospy.loginfo(f"Camera.fy: {focal_length}")
    rospy.loginfo(f"Camera.cx: {image_width / 2}")
    rospy.loginfo(f"Camera.cy: {image_height / 2}")
    rospy.loginfo(f"Camera.width: {image_width}")
    rospy.loginfo(f"Camera.height: {image_height}")
    rospy.loginfo("Camera distortion parameters (assuming no distortion):")
    rospy.loginfo("Camera.k1: 0.0")
    rospy.loginfo("Camera.k2: 0.0")
    rospy.loginfo("Camera.p1: 0.0")
    rospy.loginfo("Camera.p2: 0.0")

    expected_width, expected_height = image_width, image_height

    rate_hz = 20  # 20 Hz
    rate = rospy.Rate(rate_hz)

    seq = 0

    # Create data directory if it doesn't exist
    if not os.path.exists('./data'):
        os.makedirs('./data')

    # Remove start_time to start timestamps from zero
    # start_time = rospy.Time.now()

    try:
        while not rospy.is_shutdown():
            # Use simGetImages with compress=False to get uncompressed images
            image_request = airsim.ImageRequest(camera_name, image_type, pixels_as_float=False, compress=False)
            responses = client.simGetImages([image_request])

            if responses and len(responses) > 0 and responses[0].width != 0:
                response = responses[0]
                try:
                    # Convert AirSim image data to numpy array
                    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                    img_rgb = img1d.reshape(response.height, response.width, 3)

                    height, width = img_rgb.shape[:2]
                    rospy.loginfo(f"AirSim image dimensions: {width}x{height}")

                    # Convert to grayscale
                    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

                    # Save image to disk
                    image_filename = f'./data/image_{seq}.png'
                    cv2.imwrite(image_filename, img_gray)
                    rospy.loginfo(f"Saved image {image_filename}")

                    # Create ROS image message
                    img_msg = bridge.cv2_to_imgmsg(img_gray, encoding="mono8")

                    # Set message timestamp and frame_id
                    # Timestamps starting from zero and increasing
                    timestamp = rospy.Time.from_sec(seq * 1.0 / rate_hz)
                    img_msg.header.stamp = timestamp
                    img_msg.header.frame_id = "camera"
                    img_msg.header.seq = seq

                    # Publish image
                    image_pub.publish(img_msg)

                    rospy.loginfo(f"Published image, seq: {seq}, timestamp: {timestamp.to_sec()}")
                    seq += 1

                except Exception as e:
                    rospy.logerr(f"Error processing image: {str(e)}")
            else:
                rospy.logwarn("Failed to get valid image from AirSim")

            rate.sleep()

    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()
