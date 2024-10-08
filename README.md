# record3d_ros2_py
ROS2 package running [Record3d Sample applications](https://github.com/marek-simonik/record3d)
## Installation
~~~
# Install record3d library
# Follow Record3D repository
python -m pip install record3d

# Setup ROS environment ~  Build & Install
mkdir -p ~/ros2_ws/src
source /opt/ros/humble/setup.bash
git clone https://github.com/yuyaa199908/record3_ros2_py
cd ~/ros2_ws
colcon build --symlink-install
source ~/ros2_ws/install/setup.bash
~~~ 
## Run
~~~
ros2 launch record3d_ros2_py r3d_test.launch.py
~~~
## Topic
## Param
## TODO
- Multithreading for each topics
- Transform from pose variables to the camera coordinate system
- Publish camerainfo topic
    - [Distortion may be [0,0,0,0,0].](https://github.com/marek-simonik/record3d/issues/88#issuecomment-2282194000)