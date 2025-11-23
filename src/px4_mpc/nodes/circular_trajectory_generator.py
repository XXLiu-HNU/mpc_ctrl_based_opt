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

import rospy
import numpy as np
from mpc_msgs.msg import Trajectory, TrajectoryPoint
from geometry_msgs.msg import Point, Vector3, PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from mavros_msgs.msg import State

class CircularTrajectoryGenerator:
    """
    生成圆形飞行轨迹的类
    """
    def __init__(self):
        rospy.init_node('circular_trajectory_generator', anonymous=True)
        
        # 轨迹参数
        self.radius = rospy.get_param('~radius', 2.0)  # 圆的半径(m)
        self.height = rospy.get_param('~height', 3.0)  # 飞行高度(m)
        self.period = rospy.get_param('~period', 10.0)  # 完成一圈的时间(s)
        self.center_x = rospy.get_param('~center_x', 0.0)  # 圆心x坐标
        self.center_y = rospy.get_param('~center_y', 0.0)  # 圆心y坐标
        self.dt = rospy.get_param('~dt', 0.02)  # 轨迹点时间间隔(s)
        self.publish_rate = rospy.get_param('~publish_rate', 10.0)  # 发布频率(Hz)
        
        # 起飞等待参数
        self.wait_for_takeoff = rospy.get_param('~wait_for_takeoff', True)  # 是否等待起飞
        self.takeoff_height_threshold = rospy.get_param('~takeoff_height_threshold', 2.0)  # 起飞高度阈值(m)
        self.hover_time_threshold = rospy.get_param('~hover_time_threshold', 3.0)  # 悬停稳定时间(s)
        self.position_tolerance = rospy.get_param('~position_tolerance', 0.3)  # 位置容忍度(m)
        self.velocity_tolerance = rospy.get_param('~velocity_tolerance', 0.2)  # 速度容忍度(m/s)
        
        # 状态变量
        self.vehicle_position = np.array([0.0, 0.0, 0.0])
        self.vehicle_velocity = np.array([0.0, 0.0, 0.0])
        self.is_armed = False
        self.is_offboard = False
        self.trajectory_started = False
        self.hover_start_time = None
        
        # 从参数服务器获取里程计话题名称
        self.odom_topic = rospy.get_param('~odom_topic', '/mavros/local_position/odom')
        
        # 订阅器
        self.odom_sub = rospy.Subscriber(
            self.odom_topic, Odometry, self.odom_callback
        )
        self.state_sub = rospy.Subscriber(
            '/mavros/state', State, self.state_callback
        )
        
        # 发布器
        self.trajectory_pub = rospy.Publisher('/trajectory', Trajectory, queue_size=1)
        
        # 定时器
        self.timer = rospy.Timer(rospy.Duration(1.0/self.publish_rate), self.publish_trajectory)
        
        rospy.loginfo("=" * 60)
        rospy.loginfo("Circular trajectory generator initialized:")
        rospy.loginfo(f"  Radius: {self.radius}m")
        rospy.loginfo(f"  Height: {self.height}m")
        rospy.loginfo(f"  Period: {self.period}s")
        rospy.loginfo(f"  Center: ({self.center_x}, {self.center_y})")
        
        if self.wait_for_takeoff:
            rospy.loginfo("=" * 60)
            rospy.loginfo("WAITING FOR TAKEOFF AND HOVER:")
            rospy.loginfo(f"  Height threshold: {self.takeoff_height_threshold}m")
            rospy.loginfo(f"  Hover time: {self.hover_time_threshold}s")
            rospy.loginfo(f"  Position tolerance: {self.position_tolerance}m")
            rospy.loginfo(f"  Velocity tolerance: {self.velocity_tolerance}m/s")
            rospy.loginfo("=" * 60)
        else:
            rospy.loginfo("Trajectory will start immediately")
        
        rospy.loginfo(f"Subscribing to odometry topic: {self.odom_topic}")
    
    def odom_callback(self, msg):
        """更新无人机位置和速度（从里程计）"""
        # 更新位置
        self.vehicle_position[0] = msg.pose.pose.position.x
        self.vehicle_position[1] = msg.pose.pose.position.y
        self.vehicle_position[2] = msg.pose.pose.position.z
        
        # 重要：odom中的twist是在body坐标系中，需要转换到local坐标系
        # 使用四元数旋转将body速度转换到world坐标系
        from scipy.spatial.transform import Rotation as R
        
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
        
        self.vehicle_velocity[0] = vel_local[0]
        self.vehicle_velocity[1] = vel_local[1]
        self.vehicle_velocity[2] = vel_local[2]
    
    def state_callback(self, msg):
        """更新无人机状态"""
        self.is_armed = msg.armed
        self.is_offboard = (msg.mode == "OFFBOARD")
    
    def check_hover_condition(self):
        """检查无人机是否已经起飞并悬停稳定"""
        # 1. 检查是否解锁和OFFBOARD模式
        if not self.is_armed or not self.is_offboard:
            self.hover_start_time = None
            return False
        
        # 2. 检查高度是否达到阈值
        if self.vehicle_position[2] < self.takeoff_height_threshold:
            self.hover_start_time = None
            return False
        
        # 3. 检查位置是否稳定（速度小于阈值）
        velocity_magnitude = np.linalg.norm(self.vehicle_velocity)
        if velocity_magnitude > self.velocity_tolerance:
            self.hover_start_time = None
            return False
        
        # 4. 检查悬停时间是否足够
        current_time = rospy.Time.now()
        if self.hover_start_time is None:
            self.hover_start_time = current_time
            rospy.loginfo("Drone hovering detected, waiting for stabilization...")
            return False
        
        hover_duration = (current_time - self.hover_start_time).to_sec()
        
        # 每秒打印一次进度
        if int(hover_duration) != int(hover_duration - 0.1):  # 粗略的每秒判断
            remaining = self.hover_time_threshold - hover_duration
            if remaining > 0:
                rospy.loginfo(f"Hovering stable for {hover_duration:.1f}s, "
                            f"waiting {remaining:.1f}s more...")
        
        if hover_duration >= self.hover_time_threshold:
            return True
        
        return False
    
    def should_publish_trajectory(self):
        """判断是否应该发布轨迹"""
        # 如果已经开始发布，继续发布
        if self.trajectory_started:
            return True
        
        # 如果不需要等待起飞，直接发布
        if not self.wait_for_takeoff:
            self.trajectory_started = True
            return True
        
        # 检查悬停条件
        if self.check_hover_condition():
            self.trajectory_started = True
            rospy.loginfo("=" * 60)
            rospy.loginfo("✓ HOVER CONDITION MET!")
            rospy.loginfo(f"  Current height: {self.vehicle_position[2]:.2f}m")
            rospy.loginfo(f"  Current velocity: {np.linalg.norm(self.vehicle_velocity):.3f}m/s")
            rospy.loginfo("  Starting circular trajectory tracking...")
            rospy.loginfo("=" * 60)
            return True
        
        return False
    
    def generate_circular_trajectory(self, start_time=0.0, duration=None):
        """
        生成圆形轨迹（使用相对时间）
        """
        if duration is None:
            duration = self.period * 2  # 默认生成2圈
        
        # 计算轨迹点数量
        num_points = int(duration / self.dt) + 1
        trajectory_points = []
        
        # 角频率
        omega = 2 * np.pi / self.period
        
        for i in range(num_points):
            # 使用相对时间（从0开始）
            t_relative = i * self.dt
            if t_relative > duration:
                break
                
            # 角度
            theta = omega * t_relative
            
            # 位置
            x = self.center_x + self.radius * np.cos(theta)
            y = self.center_y + self.radius * np.sin(theta)
            z = self.height
            
            # 速度
            vx = -self.radius * omega * np.sin(theta)
            vy = self.radius * omega * np.cos(theta)
            vz = 0.0
            
            # 加速度
            ax = -self.radius * omega**2 * np.cos(theta)
            ay = -self.radius * omega**2 * np.sin(theta)
            az = 0.0
            
            # 朝向角(切线方向)
            yaw = theta + np.pi/2
            yaw_rate = omega
            
            # 创建轨迹点
            point = TrajectoryPoint()
            point.time = t_relative  # 使用相对时间
            point.position = Point(x=x, y=y, z=z)
            point.velocity = Vector3(x=vx, y=vy, z=vz)
            point.acceleration = Vector3(x=ax, y=ay, z=az)
            point.yaw = yaw
            point.yaw_rate = yaw_rate
            
            trajectory_points.append(point)
        
        return trajectory_points
    
    def publish_trajectory(self, event):
        """
        发布轨迹消息（仅在满足条件时）
        """
        # 检查是否应该发布轨迹
        if not self.should_publish_trajectory():
            return
        
        # 生成轨迹（使用相对时间，从0开始）
        points = self.generate_circular_trajectory(start_time=0.0)
        
        # 创建轨迹消息
        trajectory = Trajectory()
        trajectory.header = Header()
        trajectory.header.stamp = rospy.Time.now()
        trajectory.header.frame_id = "map"
        trajectory.points = points
        trajectory.total_duration = self.period * 2
        
        # 发布轨迹
        self.trajectory_pub.publish(trajectory)
        
        rospy.loginfo_once("✓ Publishing circular trajectory")

if __name__ == '__main__':
    try:
        generator = CircularTrajectoryGenerator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass