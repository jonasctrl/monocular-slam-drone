FROM ros:noetic

ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY=:0
ENV QT_X11_NO_MITSHM=1
ENV LANG C.UTF-8
ENV GDK_BACKEND=x11

RUN apt-get update && apt-get install -y wget gnupg2 curl && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    apt-get update

RUN apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libgtk-3-dev \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    python3-dev \
    python3-numpy \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libglew-dev \
    libboost-all-dev \
    libssl-dev \
    libeigen3-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libopenjp2-7-dev \
    libdc1394-dev \
    libepoxy-dev \
    libx11-dev \
    liblapack-dev \
    libblas-dev \
    libsuitesparse-dev \
    x11-apps \
    wget \
    unzip \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install Pangolin library
RUN mkdir -p /opt/pangolin_build && \
    cd /opt/pangolin_build && \
    git clone https://github.com/stevenlovegrove/Pangolin.git && \
    cd Pangolin && \
    mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=Release .. && \
    make -j$(nproc) && \
    make install && \
    rm -rf /opt/pangolin_build

# Install OpenCV 4.4.0 with GTK+ 3 support
RUN mkdir -p /opt/opencv_build && \
    cd /opt/opencv_build && \
    git clone https://github.com/opencv/opencv.git && \
    cd opencv && \
    git checkout 4.4.0 && \
    mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=Release \
    -D WITH_CUDA=OFF \
    -D WITH_GTK=ON \
    -D WITH_GTK_2_X=OFF \
    -D CMAKE_INSTALL_PREFIX=/usr/local .. && \
    make -j$(nproc) && \
    make install && \
    rm -rf /opt/opencv_build

# Install ORB-SLAM3
RUN mkdir -p /opt/orb_slam3 && \
    cd /opt/orb_slam3 && \
    git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git && \
    cd ORB_SLAM3 && \
    sed -i 's/++11/++14/g' CMakeLists.txt && \
    chmod +x build.sh && \
    ./build.sh

# Set up ROS environment
RUN /bin/bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash"
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc

# Set up the ROS workspace
RUN mkdir -p /opt/catkin_ws/src
COPY ./ros-wrapper /opt/catkin_ws/src/ros-wrapper
RUN cp -r /opt/orb_slam3/ORB_SLAM3/Vocabulary/ORBvoc.txt /opt/catkin_ws/src/ros-wrapper/config/ORBvoc.txt

# Install necessary dependencies for the ROS wrapper
RUN apt-get update && \
    apt-get install -y ros-${ROS_DISTRO}-cv-bridge ros-${ROS_DISTRO}-image-transport ros-${ROS_DISTRO}-tf \
    ros-${ROS_DISTRO}-hector-trajectory-server && \
    rm -rf /var/lib/apt/lists/*

# Build the workspace
RUN /bin/bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash && \
    cd /opt/catkin_ws && \
    catkin_make"

# Launch the ORB-SLAM3 ROS wrapper
WORKDIR /opt/catkin_ws/src
CMD ["/bin/bash", "-c", "source /opt/catkin_ws/devel/setup.bash && roslaunch orb_slam3_ros_wrapper euroc_mono.launch"]
