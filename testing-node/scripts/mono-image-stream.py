#!/usr/bin/env python3
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import os

def read_images(image_folder):
    image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.png')])
    return image_files

def main():
    rospy.init_node('euroc_simulator', anonymous=True)
    image_pub = rospy.Publisher('/cam0/image_raw', Image, queue_size=1)

    bridge = CvBridge()
    rate = rospy.Rate(20) 

    dataset_path = '/opt/ORB_SLAM3/Datasets/EuRoc/MH01'
    image_folder = os.path.join(dataset_path, 'mav0', 'cam0', 'data')

    image_files = read_images(image_folder)
    rospy.loginfo("Dataset playback started...")

    for img_file in image_files:
        if rospy.is_shutdown():
            break
        
        img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        img_msg = bridge.cv2_to_imgmsg(img, encoding="mono8")
        image_pub.publish(img_msg)

        rospy.loginfo(f"Published image: {img_file}")
        rate.sleep()

    rospy.loginfo("Dataset playback completed.")

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"An error occurred: {str(e)}")
