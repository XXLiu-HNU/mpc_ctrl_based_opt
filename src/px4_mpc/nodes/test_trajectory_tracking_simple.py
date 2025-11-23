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

# 添加px4_mpc包路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from px4_mpc.controllers.multirotor_trajectory_mpc import MultirotorTrajectoryMPC
from px4_mpc.models.multirotor_rate_model import MultirotorRateModel
import numpy as np

def main(use_RTI=False):
    """
    轨迹跟踪MPC测试主函数
    """
    model = MultirotorRateModel()
    mpc_controller = MultirotorTrajectoryMPC(model)
    
    print("初始化轨迹跟踪MPC控制器...")
    
    # 生成简单的圆形轨迹
    print("生成圆形轨迹...")
    trajectory_points = generate_simple_circular_trajectory()
    
    # 设置轨迹参考
    start_time = 0.0
    mpc_controller.set_trajectory_reference(trajectory_points, start_time)
    
    # 仿真参数
    x0 = mpc_controller.x0
    Tf = mpc_controller.Tf
    N_horizon = mpc_controller.N
    
    integrator = mpc_controller.integrator
    
    nx = mpc_controller.ocp_solver.acados_ocp.dims.nx
    nu = mpc_controller.ocp_solver.acados_ocp.dims.nu
    
    Nsim = 500  # 仿真步数
    dt_sim = 0.02  # 仿真时间步长
    
    simX = np.ndarray((Nsim+1, nx))
    simU = np.ndarray((Nsim, nu))
    refX = np.ndarray((Nsim+1, nx))
    
    simX[0,:] = x0
    
    t_feedback = np.zeros((Nsim))
    
    print("开始仿真...")
    
    # 闭环仿真
    for i in range(Nsim):
        current_time = i * dt_sim
        
        try:
            # 求解MPC
            input_seq, state_pred = mpc_controller.solve_trajectory(
                simX[i, :], current_time, verbose=False)
            
            simU[i,:] = input_seq[0, :]
            t_feedback[i] = mpc_controller.ocp_solver.get_stats('time_tot')
            
            # 获取参考状态用于比较
            refX[i,:] = mpc_controller.get_reference_at_time(current_time)
            
            # 仿真系统
            simX[i+1, :] = integrator.simulate(x=simX[i, :], u=simU[i,:])
            
            if i % 50 == 0:
                print(f"步数 {i}/{Nsim}, 时间: {current_time:.2f}s")
                print(f"  位置: [{simX[i,0]:.3f}, {simX[i,1]:.3f}, {simX[i,2]:.3f}]")
                print(f"  参考: [{refX[i,0]:.3f}, {refX[i,1]:.3f}, {refX[i,2]:.3f}]")
                
        except Exception as e:
            print(f"MPC求解失败 at step {i}: {e}")
            # 使用悬停控制
            hover_thrust = model.mass * 9.81
            simU[i,:] = np.array([hover_thrust, 0.0, 0.0, 0.0])
            simX[i+1, :] = integrator.simulate(x=simX[i, :], u=simU[i,:])
    
    # 获取最后一个参考状态
    refX[Nsim,:] = mpc_controller.get_reference_at_time(Nsim * dt_sim)
    
    # 计算性能指标
    t_feedback *= 1000
    print(f'\n=== 性能统计 ===')
    print(f'求解时间 (ms): 最小={np.min(t_feedback):.3f}, 中值={np.median(t_feedback):.3f}, 最大={np.max(t_feedback):.3f}')
    
    # 计算跟踪误差
    position_errors = np.linalg.norm(simX[:-1, :3] - refX[:-1, :3], axis=1)
    print(f'位置跟踪误差 (m): 平均={np.mean(position_errors):.4f}, 最大={np.max(position_errors):.4f}')
    
    # 保存结果
    try:
        import matplotlib.pyplot as plt
        
        time_vec = np.arange(Nsim + 1) * dt_sim
        
        # 绘制结果
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 位置跟踪
        axes[0, 0].plot(time_vec, simX[:, 0], 'b-', label='实际 X', linewidth=2)
        axes[0, 0].plot(time_vec, refX[:, 0], 'r--', label='参考 X', linewidth=2)
        axes[0, 0].set_ylabel('X 位置 (m)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(time_vec, simX[:, 1], 'b-', label='实际 Y', linewidth=2)
        axes[0, 1].plot(time_vec, refX[:, 1], 'r--', label='参考 Y', linewidth=2)
        axes[0, 1].set_ylabel('Y 位置 (m)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        axes[1, 0].plot(time_vec, simX[:, 2], 'b-', label='实际 Z', linewidth=2)
        axes[1, 0].plot(time_vec, refX[:, 2], 'r--', label='参考 Z', linewidth=2)
        axes[1, 0].set_ylabel('Z 位置 (m)')
        axes[1, 0].set_xlabel('时间 (s)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 3D轨迹
        axes[1, 1].plot(simX[:, 0], simX[:, 1], 'b-', label='实际轨迹', linewidth=2)
        axes[1, 1].plot(refX[:, 0], refX[:, 1], 'r--', label='参考轨迹', linewidth=2)
        axes[1, 1].plot(simX[0, 0], simX[0, 1], 'go', markersize=8, label='起点')
        axes[1, 1].set_xlabel('X 位置 (m)')
        axes[1, 1].set_ylabel('Y 位置 (m)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].axis('equal')
        
        plt.tight_layout()
        plt.savefig('/tmp/trajectory_tracking_simulation.png', dpi=150)
        print(f'\n结果图已保存至: /tmp/trajectory_tracking_simulation.png')
        
    except Exception as e:
        print(f"绘图失败: {e}")
    
    print("\n=== 仿真完成 ===")
    
    mpc_controller.ocp_solver = None

def generate_simple_circular_trajectory(radius=2.0, height=3.0, period=10.0, dt=0.02):
    """
    生成简单的圆形轨迹点（使用相对时间）
    """
    class TrajectoryPoint:
        def __init__(self):
            self.time = 0.0
            self.position = type('Position', (), {})()
            self.velocity = type('Velocity', (), {})()
            self.acceleration = type('Acceleration', (), {})()
            self.yaw = 0.0
            self.yaw_rate = 0.0
    
    duration = period * 2  # 2圈
    num_points = int(duration / dt) + 1
    trajectory_points = []
    
    omega = 2 * np.pi / period
    
    for i in range(num_points):
        t_relative = i * dt  # 相对时间，从0开始
        theta = omega * t_relative
        
        point = TrajectoryPoint()
        point.time = t_relative  # 存储相对时间
        
        # 位置
        point.position.x = radius * np.cos(theta)
        point.position.y = radius * np.sin(theta)  
        point.position.z = height
        
        # 速度
        point.velocity.x = -radius * omega * np.sin(theta)
        point.velocity.y = radius * omega * np.cos(theta)
        point.velocity.z = 0.0
        
        # 加速度
        point.acceleration.x = -radius * omega**2 * np.cos(theta)
        point.acceleration.y = -radius * omega**2 * np.sin(theta)
        point.acceleration.z = 0.0
        
        # 偏航角
        point.yaw = theta + np.pi/2
        point.yaw_rate = omega
        
        trajectory_points.append(point)
    
    return trajectory_points

if __name__ == '__main__':
    main(use_RTI=True)