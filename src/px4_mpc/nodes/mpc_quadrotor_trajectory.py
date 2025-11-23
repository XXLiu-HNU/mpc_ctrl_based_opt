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

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from px4_mpc.models.multirotor_rate_model import MultirotorRateModel
from px4_mpc.controllers.multirotor_trajectory_mpc import MultirotorTrajectoryMPC

__author__ = "Jaeyoung Lim"
__contact__ = "jalim@ethz.ch"

import rospy
import numpy as np
from scipy.spatial.transform import Rotation as R

from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray

from mavros_msgs.msg import State
from mavros_msgs.msg import AttitudeTarget
from geometry_msgs.msg import TwistStamped
from mpc_msgs.msg import Trajectory


def vector2PoseMsg(frame_id, position, attitude):
    """Convert position and attitude to PoseStamped message"""
    pose_msg = PoseStamped()
    pose_msg.header.frame_id = frame_id
    pose_msg.pose.orientation.w = attitude[0]
    pose_msg.pose.orientation.x = attitude[1]
    pose_msg.pose.orientation.y = attitude[2]
    pose_msg.pose.orientation.z = attitude[3]
    pose_msg.pose.position.x = float(position[0])
    pose_msg.pose.position.y = float(position[1])
    pose_msg.pose.position.z = float(position[2])
    return pose_msg


class QuadrotorTrajectoryMPC:
    """
    Quadrotor MPC with trajectory tracking capability
    """
    def __init__(self):
        rospy.init_node("quadrotor_trajectory_mpc")

        # Vehicle state
        self.vehicle_local_position = np.array([0.0, 0.0, 0.0])
        self.vehicle_attitude = np.array([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z]
        self.vehicle_local_velocity = np.array([0.0, 0.0, 0.0])
        
        # Default setpoint (hover position)
        self.setpoint_position = np.array([0.0, 0.0, 3.0])
        
        # Trajectory tracking
        self.trajectory_active = False
        self.trajectory_start_time = None
        
        # Data recording for analysis
        self.record_data = rospy.get_param('~record_data', True)
        self.recorded_states = []
        self.recorded_references = []
        self.recorded_times = []
        self.recording_start_time = None
        
        # 从参数服务器获取里程计话题名称
        self.odom_topic = rospy.get_param('~odom_topic', '/mavros/local_position/odom')

        # ROS subscribers
        self.status_sub = rospy.Subscriber(
            "/mavros/state", State, self.vehicle_status_callback
        )
        self.odom_sub = rospy.Subscriber(
            self.odom_topic,
            Odometry,
            self.vehicle_odom_callback,
        )
        self.trajectory_sub = rospy.Subscriber(
            "/trajectory", Trajectory, self.trajectory_callback
        )

        # ROS publishers
        self.publisher_rates_setpoint = rospy.Publisher(
            "/mavros/setpoint_raw/attitude", AttitudeTarget, queue_size=1
        )
        self.predicted_path_pub = rospy.Publisher(
            "/px4_mpc/predicted_path", Path, queue_size=10
        )
        self.reference_path_pub = rospy.Publisher(
            "/px4_mpc/reference_path", Path, queue_size=10
        )
        self.reference_marker_pub = rospy.Publisher(
            "/px4_mpc/reference_marker", Marker, queue_size=10
        )

        self.nav_state = State.MODE_PX4_READY

        # Create model and MPC controller
        self.model = MultirotorRateModel()
        self.mpc = MultirotorTrajectoryMPC(self.model)

        rospy.loginfo("Quadrotor Trajectory MPC initialized")
        rospy.loginfo("  Prediction horizon: {}".format(self.mpc.N))
        rospy.loginfo("  Prediction time: {} s".format(self.mpc.Tf))
        rospy.loginfo("  Odometry topic: {}".format(self.odom_topic))

        # Control loop timer (50 Hz)
        timer_period = 0.02
        self.timer = rospy.Timer(rospy.Duration(timer_period), self.cmdloop_callback)

    def trajectory_callback(self, msg):
        """Handle incoming trajectory messages"""
        rospy.loginfo("Received trajectory with {} points".format(len(msg.points)))
        
        if len(msg.points) > 0:
            # 只在第一次接收轨迹或轨迹未激活时设置起始时间
            if not self.trajectory_active:
                self.trajectory_start_time = rospy.Time.now().to_sec()
                self.mpc.set_trajectory_reference(msg.points, self.trajectory_start_time)
                self.trajectory_active = True
                rospy.loginfo("Trajectory tracking activated at time: {:.2f}".format(
                    self.trajectory_start_time))
            else:
                # 轨迹已激活，只更新轨迹点，保持原始起始时间
                self.mpc.trajectory_reference = msg.points
                rospy.loginfo_throttle(5.0, "Trajectory updated (keeping start time)")

    def vehicle_odom_callback(self, msg):
        """Update vehicle state from odometry"""
        # 更新位置
        self.vehicle_local_position[0] = msg.pose.pose.position.x
        self.vehicle_local_position[1] = msg.pose.pose.position.y
        self.vehicle_local_position[2] = msg.pose.pose.position.z

        # 更新姿态四元数 [w, x, y, z]
        self.vehicle_attitude[0] = msg.pose.pose.orientation.w
        self.vehicle_attitude[1] = msg.pose.pose.orientation.x
        self.vehicle_attitude[2] = msg.pose.pose.orientation.y
        self.vehicle_attitude[3] = msg.pose.pose.orientation.z
        
        # 重要：odom中的twist是在body坐标系中，需要转换到local坐标系
        # 使用四元数旋转将body速度转换到world坐标系
        quat = R.from_quat([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ])
        
        # body坐标系的速度
        vel_body = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z
        ])
        
        # 转换到world/local坐标系
        vel_local = quat.apply(vel_body)
        
        self.vehicle_local_velocity[0] = vel_local[0]
        self.vehicle_local_velocity[1] = vel_local[1]
        self.vehicle_local_velocity[2] = vel_local[2]

    def vehicle_status_callback(self, msg):
        """Update vehicle flight mode"""
        self.nav_state = msg.mode

    def publish_reference_marker(self, pub, position):
        """Publish current reference position as a marker"""
        msg = Marker()
        msg.action = Marker.ADD
        msg.header.frame_id = "map"
        msg.header.stamp = rospy.Time.now()
        msg.ns = "reference"
        msg.id = 1
        msg.type = Marker.SPHERE
        msg.scale.x = 0.5
        msg.scale.y = 0.5
        msg.scale.z = 0.5
        msg.color.r = 1.0
        msg.color.g = 0.0
        msg.color.b = 0.0
        msg.color.a = 0.8
        msg.pose.position.x = position[0]
        msg.pose.position.y = position[1]
        msg.pose.position.z = position[2]
        msg.pose.orientation.w = 1.0
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0

        pub.publish(msg)

    def publish_reference_path(self, pub, reference_states):
        """Publish reference trajectory path"""
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = rospy.Time.now()
        
        for ref_state in reference_states:
            pose_msg = vector2PoseMsg(
                "map",
                ref_state[0:3],  # position
                ref_state[6:10]  # quaternion
            )
            path_msg.poses.append(pose_msg)
        
        pub.publish(path_msg)

    def cmdloop_callback(self, timer):
        """Main control loop callback"""
        try:
            current_time = rospy.Time.now().to_sec()
            
            # Build current state vector
            x0 = np.array([
                self.vehicle_local_position[0],
                self.vehicle_local_position[1],
                self.vehicle_local_position[2],
                self.vehicle_local_velocity[0],
                self.vehicle_local_velocity[1],
                self.vehicle_local_velocity[2],
                self.vehicle_attitude[0],
                self.vehicle_attitude[1],
                self.vehicle_attitude[2],
                self.vehicle_attitude[3],
            ])

            if self.trajectory_active and self.mpc.trajectory_reference is not None:
                # Trajectory tracking mode
                try:
                    u_pred, x_pred = self.mpc.solve_trajectory(x0, current_time, verbose=False)
                    
                    # Get current reference for visualization
                    current_ref = self.mpc.get_reference_at_time(current_time)
                    
                    # Collect reference trajectory for visualization
                    dt = self.mpc.Tf / self.mpc.N
                    ref_states = []
                    for i in range(self.mpc.N + 1):
                        future_time = current_time + i * dt
                        ref_state = self.mpc.get_reference_at_time(future_time)
                        ref_states.append(ref_state)
                    
                    # Publish reference path
                    self.publish_reference_path(self.reference_path_pub, ref_states)
                    self.publish_reference_marker(self.reference_marker_pub, current_ref[0:3])
                    
                    # Record data for analysis
                    if self.record_data:
                        if self.recording_start_time is None:
                            self.recording_start_time = current_time
                        self.recorded_times.append(current_time - self.recording_start_time)
                        self.recorded_states.append(x0.copy())
                        self.recorded_references.append(current_ref.copy())
                    
                except Exception as e:
                    rospy.logwarn_throttle(1.0, "MPC solve failed: {}".format(e))
                    # Fall back to hover
                    u_pred = np.array([[self.model.mass * 9.81, 0.0, 0.0, 0.0]])
                    x_pred = np.tile(x0, (self.mpc.N + 1, 1))
            else:
                # Point-to-point mode (original behavior)
                error_position = self.vehicle_local_position - self.setpoint_position
                
                x0_error = np.array([
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
                ])
                
                try:
                    u_pred, x_pred = self.mpc.solve(x0_error)
                    # Adjust predicted states back to world frame
                    x_pred[:, 0:3] = x_pred[:, 0:3] + self.setpoint_position
                except Exception as e:
                    rospy.logwarn_throttle(1.0, "MPC solve failed: {}".format(e))
                    u_pred = np.array([[self.model.mass * 9.81, 0.0, 0.0, 0.0]])
                    x_pred = np.tile(x0, (self.mpc.N + 1, 1))
                
                self.publish_reference_marker(self.reference_marker_pub, self.setpoint_position)

            # Publish predicted path
            predicted_path_msg = Path()
            predicted_path_msg.header.frame_id = "map"
            predicted_path_msg.header.stamp = rospy.Time.now()
            
            for predicted_state in x_pred:
                predicted_pose_msg = vector2PoseMsg(
                    "map",
                    predicted_state[0:3],
                    predicted_state[6:10]
                )
                predicted_path_msg.poses.append(predicted_pose_msg)
            
            self.predicted_path_pub.publish(predicted_path_msg)

            # Extract control commands
            thrust_rates = u_pred[0, :]
            # Scale thrust command (hover thrust = 0.73)
            thrust_command = thrust_rates[0] * 0.073

            # Publish control setpoint
            setpoint_msg = AttitudeTarget()
            setpoint_msg.header.stamp = rospy.Time.now()
            setpoint_msg.type_mask = AttitudeTarget.IGNORE_ATTITUDE
            setpoint_msg.body_rate.x = float(thrust_rates[1])
            setpoint_msg.body_rate.y = float(thrust_rates[2])
            setpoint_msg.body_rate.z = float(thrust_rates[3])
            setpoint_msg.thrust = float(thrust_command)
            self.publisher_rates_setpoint.publish(setpoint_msg)

        except Exception as e:
            rospy.logerr("Control loop error: {}".format(e))
    
    def save_and_plot_results(self):
        """保存数据并绘制结果"""
        if not self.record_data or len(self.recorded_times) == 0:
            rospy.loginfo("No data recorded for analysis")
            return
        
        try:
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
            
            times = np.array(self.recorded_times)
            states = np.array(self.recorded_states)
            references = np.array(self.recorded_references)
            
            # 计算位置误差
            position_errors = np.linalg.norm(states[:, 0:3] - references[:, 0:3], axis=1)
            
            # 计算RMSE
            rmse_x = np.sqrt(np.mean((states[:, 0] - references[:, 0])**2))
            rmse_y = np.sqrt(np.mean((states[:, 1] - references[:, 1])**2))
            rmse_z = np.sqrt(np.mean((states[:, 2] - references[:, 2])**2))
            rmse_total = np.sqrt(np.mean(position_errors**2))
            
            rospy.loginfo("=" * 60)
            rospy.loginfo("TRAJECTORY TRACKING ANALYSIS")
            rospy.loginfo("=" * 60)
            rospy.loginfo("Total samples: {}".format(len(times)))
            rospy.loginfo("Duration: {:.2f}s".format(times[-1]))
            rospy.loginfo("")
            rospy.loginfo("RMSE:")
            rospy.loginfo("  X: {:.4f} m".format(rmse_x))
            rospy.loginfo("  Y: {:.4f} m".format(rmse_y))
            rospy.loginfo("  Z: {:.4f} m".format(rmse_z))
            rospy.loginfo("  Total: {:.4f} m".format(rmse_total))
            rospy.loginfo("")
            rospy.loginfo("Position Error:")
            rospy.loginfo("  Mean: {:.4f} m".format(np.mean(position_errors)))
            rospy.loginfo("  Max:  {:.4f} m".format(np.max(position_errors)))
            rospy.loginfo("  Std:  {:.4f} m".format(np.std(position_errors)))
            rospy.loginfo("=" * 60)
            
            # 创建图形
            fig = plt.figure(figsize=(16, 10))
            gs = GridSpec(3, 3, figure=fig)
            
            # 3D轨迹对比
            ax1 = fig.add_subplot(gs[0:2, 0:2], projection='3d')
            ax1.plot(states[:, 0], states[:, 1], states[:, 2], 'b-', linewidth=2, label='Actual')
            ax1.plot(references[:, 0], references[:, 1], references[:, 2], 'r--', linewidth=2, label='Reference')
            ax1.plot([states[0, 0]], [states[0, 1]], [states[0, 2]], 'go', markersize=10, label='Start')
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_zlabel('Z (m)')
            ax1.set_title('3D Trajectory Comparison', fontsize=12, fontweight='bold')
            ax1.legend()
            ax1.grid(True)
            
            # XY平面轨迹
            ax2 = fig.add_subplot(gs[0, 2])
            ax2.plot(states[:, 0], states[:, 1], 'b-', linewidth=2, label='Actual')
            ax2.plot(references[:, 0], references[:, 1], 'r--', linewidth=2, label='Reference')
            ax2.plot(states[0, 0], states[0, 1], 'go', markersize=8, label='Start')
            ax2.set_xlabel('X (m)')
            ax2.set_ylabel('Y (m)')
            ax2.set_title('XY Plane View')
            ax2.legend(fontsize=8)
            ax2.grid(True)
            ax2.axis('equal')
            
            # 位置误差时间历程
            ax3 = fig.add_subplot(gs[1, 2])
            ax3.plot(times, position_errors, 'b-', linewidth=1.5)
            ax3.axhline(y=rmse_total, color='r', linestyle='--', label='RMSE: {:.3f}m'.format(rmse_total))
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Position Error (m)')
            ax3.set_title('Position Error vs Time')
            ax3.legend(fontsize=8)
            ax3.grid(True)
            
            # X位置对比
            ax4 = fig.add_subplot(gs[2, 0])
            ax4.plot(times, states[:, 0], 'b-', linewidth=2, label='Actual')
            ax4.plot(times, references[:, 0], 'r--', linewidth=2, label='Reference')
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('X Position (m)')
            ax4.set_title('X Position Tracking (RMSE: {:.3f}m)'.format(rmse_x))
            ax4.legend()
            ax4.grid(True)
            
            # Y位置对比
            ax5 = fig.add_subplot(gs[2, 1])
            ax5.plot(times, states[:, 1], 'b-', linewidth=2, label='Actual')
            ax5.plot(times, references[:, 1], 'r--', linewidth=2, label='Reference')
            ax5.set_xlabel('Time (s)')
            ax5.set_ylabel('Y Position (m)')
            ax5.set_title('Y Position Tracking (RMSE: {:.3f}m)'.format(rmse_y))
            ax5.legend()
            ax5.grid(True)
            
            # Z位置对比
            ax6 = fig.add_subplot(gs[2, 2])
            ax6.plot(times, states[:, 2], 'b-', linewidth=2, label='Actual')
            ax6.plot(times, references[:, 2], 'r--', linewidth=2, label='Reference')
            ax6.set_xlabel('Time (s)')
            ax6.set_ylabel('Z Position (m)')
            ax6.set_title('Z Position Tracking (RMSE: {:.3f}m)'.format(rmse_z))
            ax6.legend()
            ax6.grid(True)
            
            plt.tight_layout()
            
            # 保存图片
            save_path = '/tmp/trajectory_tracking_results.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            rospy.loginfo("Results saved to: {}".format(save_path))
            
            # 保存CSV数据
            csv_path = '/tmp/trajectory_tracking_data.csv'
            with open(csv_path, 'w') as f:
                f.write('time,actual_x,actual_y,actual_z,ref_x,ref_y,ref_z,error\n')
                for i in range(len(times)):
                    f.write('{},{},{},{},{},{},{},{}\n'.format(
                        times[i],
                        states[i, 0], states[i, 1], states[i, 2],
                        references[i, 0], references[i, 1], references[i, 2],
                        position_errors[i]
                    ))
            rospy.loginfo("Data saved to: {}".format(csv_path))
            
        except Exception as e:
            rospy.logerr("Failed to save/plot results: {}".format(e))
            import traceback
            traceback.print_exc()


def main(args=None):
    try:
        quadrotor_mpc = QuadrotorTrajectoryMPC()
        
        # 设置关闭回调
        def shutdown_hook():
            rospy.loginfo("Shutting down, analyzing trajectory tracking performance...")
            quadrotor_mpc.save_and_plot_results()
        
        rospy.on_shutdown(shutdown_hook)
        
        rospy.loginfo("Quadrotor Trajectory MPC node started")
        rospy.loginfo("Ready to receive trajectory on /trajectory topic")
        rospy.loginfo("Press Ctrl+C to stop and generate analysis")
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr("Node failed: {}".format(e))


if __name__ == "__main__":
    main()
