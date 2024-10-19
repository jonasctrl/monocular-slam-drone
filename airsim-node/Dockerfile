FROM ros:noetic-ros-core

ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY=:0
ENV QT_X11_NO_MITSHM=1
ENV LANG C.UTF-8
ENV GDK_BACKEND=x11

# Install essential dependencies
RUN apt-get update && \
    apt-get install -y \
    python3-pip \
    ros-noetic-catkin \
    ros-noetic-tf2-ros \
    ros-noetic-visualization-msgs \
    ros-noetic-sensor-msgs \
    ros-noetic-nav-msgs \
    ros-noetic-geometry-msgs \
    ros-noetic-cv-bridge \
    ros-noetic-image-transport \
    ros-noetic-camera-info-manager \
    ros-noetic-roslint \
    ros-noetic-rosbash \
    python3-catkin-tools \
    libgl1-mesa-glx \
    git \
    build-essential \
    cmake \
    libboost-all-dev \
    libpython3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up AirSim and Python dependencies
RUN pip3 install msgpack-rpc-python
RUN pip3 install airsim

# Create and set up catkin workspace
RUN mkdir -p /catkin_ws/src
WORKDIR /catkin_ws/src

# Clone airsim_ros_client repository
RUN git clone https://github.com/Mayankm96/airsim_ros_client.git

# Install python3-rosdep and initialize rosdep
RUN apt-get update && \
    apt-get install -y python3-rosdep && \
    rosdep init && \
    rosdep update

# Install dependencies with rosdep
WORKDIR /catkin_ws
RUN rosdep install --from-paths src --ignore-src -r -y

# Build the workspace
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make"

# Source environment on container start and set ROS_PACKAGE_PATH
RUN echo "source /catkin_ws/devel/setup.bash" >> ~/.bashrc
ENV ROS_PACKAGE_PATH=/catkin_ws/src

COPY ./scripts /opt/
WORKDIR /opt/scripts

CMD ["sleep", "infinity"]
