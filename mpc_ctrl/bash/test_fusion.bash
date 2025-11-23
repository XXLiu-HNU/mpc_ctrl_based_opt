#!/bin/bash
echo -e " ${RED} [ Running mpc_ctrl nodes ] ${NC}"
source /home/amov/Fast250/devel/setup.bash

roslaunch mpc_ctrl vicon.launch & sleep 1;

echo -e " ${RED} [ Running fusion nodes ] ${NC}" 
source /home/amov/Fast250/devel/setup.bash

roslaunch mpc_ctrl run_fusion.launch & sleep 1;

wait;
