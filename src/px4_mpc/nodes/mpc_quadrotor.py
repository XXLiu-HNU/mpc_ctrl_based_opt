#!/usr/bin/env python3

############################################################################
#
#   Copyright (C) 2023 PX4 Development Team. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
# 3. Neither the name PX4 nor the names of its contributors may be
#    used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
############################################################################

from px4_mpc.models.multirotor_rate_model import MultirotorRateModel
from px4_mpc.controllers.multirotor_rate_mpc import MultirotorRateMPC

__author__ = "Jaeyoung Lim"
__contact__ = "jalim@ethz.ch"

import rospy
import numpy as np

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker

from mavros_msgs.msg import State
from mavros_msgs.msg import AttitudeTarget
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker

# from mpc_msgs.srv import SetPose


def vector2PoseMsg(frame_id, position, attitude):
    pose_msg = PoseStamped()
    # msg.header.stamp = Clock().now().nanoseconds / 1000
    pose_msg.header.frame_id = frame_id
    pose_msg.pose.orientation.w = attitude[0]
    pose_msg.pose.orientation.x = attitude[1]
    pose_msg.pose.orientation.y = attitude[2]
    pose_msg.pose.orientation.z = attitude[3]
    pose_msg.pose.position.x = float(position[0])
    pose_msg.pose.position.y = float(position[1])
    pose_msg.pose.position.z = float(position[2])
    return pose_msg


class QuadrotorMPC:
    def __init__(self):
        rospy.init_node("minimal_publisher")

        self.vehicle_local_position = np.array([0.0, 0.0, 0.0])
        self.vehicle_attitude = np.array([1.0, 0.0, 0.0, 0.0])
        self.vehicle_local_velocity = np.array([0.0, 0.0, 0.0])
        self.setpoint_position = np.array([0.0, 0.0, 3.0])

        self.status_sub = rospy.Subscriber(
            "/mavros/state", State, self.vehicle_status_callback
        )

        self.local_twist_sub = rospy.Subscriber(
            "/mavros/local_position/velocity_local",
            TwistStamped,
            self.vehicle_local_twist_callback,
        )

        self.local_position_sub = rospy.Subscriber(
            "/mavros/local_position/pose",
            PoseStamped,
            self.vehicle_local_position_callback,
        )

        # self.set_pose_srv = self.create_service(SetPose, '/set_pose', self.add_set_pos_callback)

        # self.publisher_offboard_mode = rospy.Publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.publisher_rates_setpoint = rospy.Publisher(
            "/mavros/setpoint_raw/attitude", AttitudeTarget, queue_size=1
        )
        self.predicted_path_pub = rospy.Publisher(
            "/px4_mpc/predicted_path", Path, queue_size=10
        )
        self.reference_pub = rospy.Publisher(
            "/px4_mpc/reference", Marker, queue_size=10
        )

        self.nav_state = State.MODE_PX4_READY

        # Create Quadrotor and controller objects
        self.model = MultirotorRateModel()

        # Create MPC Solver
        MPC_HORIZON = 15

        # Spawn Controller
        self.mpc = MultirotorRateMPC(self.model)

        timer_period = 0.02  # seconds
        self.timer = rospy.Timer(rospy.Duration(timer_period), self.cmdloop_callback)

    def vehicle_local_twist_callback(self, msg):
        self.vehicle_local_velocity[0] = msg.twist.linear.x
        self.vehicle_local_velocity[1] = msg.twist.linear.y
        self.vehicle_local_velocity[2] = msg.twist.linear.z

    def vehicle_local_position_callback(self, msg):
        self.vehicle_local_position[0] = msg.pose.position.x
        self.vehicle_local_position[1] = msg.pose.position.y
        self.vehicle_local_position[2] = msg.pose.position.z

        # # TODO: handle NED->ENU transformation
        self.vehicle_attitude[0] = msg.pose.orientation.w
        self.vehicle_attitude[1] = msg.pose.orientation.x
        self.vehicle_attitude[2] = msg.pose.orientation.y
        self.vehicle_attitude[3] = msg.pose.orientation.z

    def vehicle_status_callback(self, msg):
        self.nav_state = msg.mode

    def publish_reference(self, pub, reference):
        msg = Marker()
        msg.action = Marker.ADD
        msg.header.frame_id = "map"
        # msg.header.stamp = Clock().now().nanoseconds / 1000
        msg.ns = "arrow"
        msg.id = 1
        msg.type = Marker.SPHERE
        msg.scale.x = 0.5
        msg.scale.y = 0.5
        msg.scale.z = 0.5
        msg.color.r = 1.0
        msg.color.g = 0.0
        msg.color.b = 0.0
        msg.color.a = 1.0
        msg.pose.position.x = reference[0]
        msg.pose.position.y = reference[1]
        msg.pose.position.z = reference[2]
        msg.pose.orientation.w = 1.0
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0

        pub.publish(msg)

    def cmdloop_callback(self, timer):
        # Publish offboard control modes
        # offboard_msg = OffboardControlMode()
        # offboard_msg.timestamp = int(Clock().now().nanoseconds / 1000)
        # offboard_msg.position = False
        # offboard_msg.velocity = False
        # offboard_msg.acceleration = False
        # offboard_msg.attitude = False
        # offboard_msg.body_rate = True
        # self.publisher_offboard_mode.publish(offboard_msg)

        error_position = self.vehicle_local_position - self.setpoint_position

        x0 = np.array(
            [
                error_position[0],
                error_position[1],
                error_position[2],
                self.vehicle_local_velocity[0],
                self.vehicle_local_velocity[1],
                self.vehicle_local_velocity[2],
                self.vehicle_attitude[0],
                self.vehicle_attitude[1],
                self.vehicle_attitude[2],
                self.vehicle_attitude[3],
            ]
        ).reshape(10, 1)

        u_pred, x_pred = self.mpc.solve(x0)

        idx = 0
        predicted_path_msg = Path()
        for predicted_state in x_pred:
            idx = idx + 1
            # Publish time history of the vehicle path
            predicted_pose_msg = vector2PoseMsg(
                "map",
                predicted_state[0:3] + self.setpoint_position,
                np.array([1.0, 0.0, 0.0, 0.0]),
            )
            predicted_path_msg.header = predicted_pose_msg.header
            predicted_path_msg.poses.append(predicted_pose_msg)
        self.predicted_path_pub.publish(predicted_path_msg)
        self.publish_reference(self.reference_pub, self.setpoint_position)

        thrust_rates = u_pred[0, :]
        # Hover thrust = 0.73
        thrust_command = (thrust_rates[0] * 0.073 + 0.0)
        # if self.nav_state == State.MODE_PX4_OFFBOARD:
        setpoint_msg = AttitudeTarget()
        setpoint_msg.header.stamp = rospy.Time.now()
        setpoint_msg.type_mask = AttitudeTarget.IGNORE_ATTITUDE
        setpoint_msg.body_rate.x = float(thrust_rates[1])
        setpoint_msg.body_rate.y = float(thrust_rates[2])
        setpoint_msg.body_rate.z = float(thrust_rates[3])
        setpoint_msg.thrust = float(thrust_command)
        self.publisher_rates_setpoint.publish(setpoint_msg)

    def add_set_pos_callback(self, request, response):
        self.setpoint_position[0] = request.pose.position.x
        self.setpoint_position[1] = request.pose.position.y
        self.setpoint_position[2] = request.pose.position.z

        return response


def main(args=None):
    quadrotor_mpc = QuadrotorMPC()
    rospy.spin()


if __name__ == "__main__":
    main()
