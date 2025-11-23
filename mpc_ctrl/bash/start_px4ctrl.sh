#!/bin/bash

echo -e " ${RED} [ 7. Running JPCM nodes ] ${NC}" 
source /home/amov/Fast250/devel/setup.bash

roslaunch mpc_ctrl run_ctrl.launch & sleep 5;

wait;