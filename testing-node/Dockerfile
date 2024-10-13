FROM ros:noetic-ros-base

RUN apt-get update && apt-get install -y \
    python3-pip \
    ros-noetic-cv-bridge \
    ros-noetic-tf \
    python3-opencv \
    wget \
    vim \
    unzip

RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc

RUN mkdir -p /opt/ORB_SLAM3/Datasets/EuRoc && \
    cd /opt/ORB_SLAM3/Datasets/EuRoc/ && \
    wget -c http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/MH_01_easy.zip && \
    mkdir MH01 && unzip MH_01_easy.zip -d MH01/ && \
    rm MH_01_easy.zip

ENTRYPOINT ["/ros_entrypoint.sh"]

CMD ["bash", "-c", "source /opt/ros/${ROS_DISTRO}/setup.bash && python3 /opt/scripts/mono-image-stream.py"]
