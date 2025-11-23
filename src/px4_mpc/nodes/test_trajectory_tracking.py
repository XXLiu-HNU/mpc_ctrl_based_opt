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
# COPYRIGHT OWNER OR CONTRIBUTORS be LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
############################################################################

import rospy
import numpy as np
from px4_mpc.controllers.multirotor_trajectory_mpc import MultirotorTrajectoryMPC
from px4_mpc.models.multirotor_rate_model import MultirotorRateModel
from px4_mpc.visualization import plot_multirotor
from mpc_msgs.msg import Trajectory
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
import threading

class TrajectoryTrackingNode:
    """
    轨迹跟踪ROS节点
    """
    
    def __init__(self):
        rospy.init_node('trajectory_tracking_mpc', anonymous=True)
        
        # 初始化MPC控制器
        self.model = MultirotorRateModel()
        self.mpc_controller = MultirotorTrajectoryMPC(self.model)
        
        # 当前状态
        self.current_state = self.mpc_controller.x0.copy()
        self.state_lock = threading.Lock()
        
        # 仿真相关
        self.integrator = self.mpc_controller.integrator
        self.simulation_dt = 0.02  # 20ms
        
        # ROS 订阅者和发布者
        self.trajectory_sub = rospy.Subscriber('/trajectory', Trajectory, self.trajectory_callback)
        self.pose_pub = rospy.Publisher('/mavros/setpoint_position/local', PoseStamped, queue_size=1)
        self.velocity_pub = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel', TwistStamped, queue_size=1)
        self.odom_pub = rospy.Publisher('/odom', Odometry, queue_size=1)
        
        # 控制循环定时器
        self.control_timer = rospy.Timer(rospy.Duration(self.simulation_dt), self.control_loop)
        
        # 仿真数据记录
        self.simulation_data = {
            'time': [],
            'states': [],
            'controls': [],
            'references': []
        }
        self.simulation_start_time = rospy.Time.now().to_sec()
        
        rospy.loginfo("Trajectory tracking MPC node initialized")
        
    def trajectory_callback(self, msg):
        """
        接收轨迹消息并设置MPC参考
        """
        rospy.loginfo(f"Received trajectory with {len(msg.points)} points")
        
        # 设置轨迹参考
        current_time = rospy.Time.now().to_sec()
        self.mpc_controller.set_trajectory_reference(msg.points, current_time)
        
        rospy.loginfo("Trajectory reference set successfully")
    
    def control_loop(self, event):
        """
        主控制循环
        """
        try:
            current_time = rospy.Time.now().to_sec()
            
            with self.state_lock:
                current_state = self.current_state.copy()
            
            # 只有在有轨迹参考时才进行控制
            if self.mpc_controller.trajectory_reference is not None:
                # 求解MPC
                try:
                    simU, simX = self.mpc_controller.solve_trajectory(
                        current_state, current_time, verbose=False)
                    
                    # 获取第一个控制输入
                    u_current = simU[0, :]
                    
                    # 仿真系统动态
                    next_state = self.integrator.simulate(x=current_state, u=u_current)
                    
                    with self.state_lock:
                        self.current_state = next_state
                    
                    # 发布控制指令和状态信息
                    self.publish_control_commands(u_current)
                    self.publish_state_info(current_state, current_time)
                    
                    # 记录仿真数据
                    ref_state = self.mpc_controller.get_reference_at_time(current_time)
                    self.record_simulation_data(current_time, current_state, u_current, ref_state)
                    
                except Exception as e:
                    rospy.logwarn(f"MPC solve failed: {e}")
            else:
                # 悬停控制
                self.hover_control()
                
        except Exception as e:
            rospy.logerr(f"Control loop error: {e}")
    
    def hover_control(self):
        """
        悬停控制逻辑
        """
        # 简单的悬停控制：保持当前位置，零速度
        hover_thrust = self.model.mass * 9.81  # 悬停推力
        hover_input = np.array([hover_thrust, 0.0, 0.0, 0.0])
        
        # 仿真系统动态
        with self.state_lock:
            next_state = self.integrator.simulate(x=self.current_state, u=hover_input)
            self.current_state = next_state
        
        # 发布悬停指令
        self.publish_control_commands(hover_input)
    
    def publish_control_commands(self, u):
        """
        发布控制指令到PX4
        """
        # 这里可以根据需要发布到不同的话题
        # 目前只是示例，实际使用时需要根据PX4接口调整
        pass
    
    def publish_state_info(self, state, timestamp):
        """
        发布状态信息
        """
        # 发布里程计信息
        odom_msg = Odometry()
        odom_msg.header.stamp = rospy.Time.from_sec(timestamp)
        odom_msg.header.frame_id = "map"
        odom_msg.child_frame_id = "base_link"
        
        # 位置
        odom_msg.pose.pose.position.x = state[0]
        odom_msg.pose.pose.position.y = state[1]
        odom_msg.pose.pose.position.z = state[2]
        
        # 姿态(四元数)
        odom_msg.pose.pose.orientation.w = state[6]
        odom_msg.pose.pose.orientation.x = state[7]
        odom_msg.pose.pose.orientation.y = state[8]
        odom_msg.pose.pose.orientation.z = state[9]
        
        # 速度
        odom_msg.twist.twist.linear.x = state[3]
        odom_msg.twist.twist.linear.y = state[4]
        odom_msg.twist.twist.linear.z = state[5]
        
        self.odom_pub.publish(odom_msg)
    
    def record_simulation_data(self, time, state, control, reference):
        """
        记录仿真数据用于分析
        """
        self.simulation_data['time'].append(time - self.simulation_start_time)
        self.simulation_data['states'].append(state.copy())
        self.simulation_data['controls'].append(control.copy())
        self.simulation_data['references'].append(reference.copy())
        
        # 限制数据长度以避免内存问题
        max_length = 10000
        if len(self.simulation_data['time']) > max_length:
            for key in self.simulation_data:
                self.simulation_data[key] = self.simulation_data[key][-max_length:]
    
    def shutdown(self):
        """
        节点关闭时的清理工作
        """
        rospy.loginfo("Shutting down trajectory tracking node")
        
        # 可选：保存仿真数据
        if len(self.simulation_data['time']) > 0:
            try:
                import matplotlib.pyplot as plt
                
                times = np.array(self.simulation_data['time'])
                states = np.array(self.simulation_data['states'])
                references = np.array(self.simulation_data['references'])
                
                # 绘制轨迹跟踪结果
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                
                # 位置跟踪
                axes[0, 0].plot(times, states[:, 0], 'b-', label='Actual X')
                axes[0, 0].plot(times, references[:, 0], 'r--', label='Reference X')
                axes[0, 0].set_ylabel('X Position (m)')
                axes[0, 0].legend()
                axes[0, 0].grid(True)
                
                axes[0, 1].plot(times, states[:, 1], 'b-', label='Actual Y')
                axes[0, 1].plot(times, references[:, 1], 'r--', label='Reference Y')
                axes[0, 1].set_ylabel('Y Position (m)')
                axes[0, 1].legend()
                axes[0, 1].grid(True)
                
                axes[1, 0].plot(times, states[:, 2], 'b-', label='Actual Z')
                axes[1, 0].plot(times, references[:, 2], 'r--', label='Reference Z')
                axes[1, 0].set_ylabel('Z Position (m)')
                axes[1, 0].set_xlabel('Time (s)')
                axes[1, 0].legend()
                axes[1, 0].grid(True)
                
                # 3D轨迹
                axes[1, 1].plot(states[:, 0], states[:, 1], 'b-', label='Actual')
                axes[1, 1].plot(references[:, 0], references[:, 1], 'r--', label='Reference')
                axes[1, 1].set_xlabel('X Position (m)')
                axes[1, 1].set_ylabel('Y Position (m)')
                axes[1, 1].legend()
                axes[1, 1].grid(True)
                axes[1, 1].axis('equal')
                
                plt.tight_layout()
                plt.savefig('/tmp/trajectory_tracking_result.png')
                rospy.loginfo("Trajectory tracking result saved to /tmp/trajectory_tracking_result.png")
                
            except Exception as e:
                rospy.logwarn(f"Failed to save tracking results: {e}")

def main():
    try:
        node = TrajectoryTrackingNode()
        
        # 设置关闭回调
        rospy.on_shutdown(node.shutdown)
        
        rospy.loginfo("Trajectory tracking MPC node started. Waiting for trajectory...")
        rospy.spin()
        
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Node failed: {e}")

if __name__ == '__main__':
    main()