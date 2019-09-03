#!/bin/sh
source /opt/ros/kinetic/setup.bash;
sudo chmod a+rw /dev/ttyACM0;
rosrun urg_node urg_node