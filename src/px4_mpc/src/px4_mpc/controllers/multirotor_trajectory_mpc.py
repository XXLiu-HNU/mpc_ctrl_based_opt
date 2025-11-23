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

from px4_mpc.controllers.multirotor_rate_mpc import MultirotorRateMPC
import numpy as np
from scipy.spatial.transform import Rotation as R

class MultirotorTrajectoryMPC(MultirotorRateMPC):
    """
    扩展原有的MPC控制器以支持轨迹跟踪
    """
    
    def __init__(self, model):
        super().__init__(model)
        # 当前轨迹参考
        self.trajectory_reference = None
        self.trajectory_start_time = None
        
    def set_trajectory_reference(self, trajectory_points, start_time):
        """
        设置轨迹参考
        
        Args:
            trajectory_points: 轨迹点列表，每个点包含时间、位置、速度、加速度、偏航角等
            start_time: 轨迹开始时间
        """
        self.trajectory_reference = trajectory_points
        self.trajectory_start_time = start_time
        
    def get_reference_at_time(self, t):
        """
        获取指定时间的参考状态
        
        Args:
            t: 当前绝对时间戳
            
        Returns:
            reference_state: 参考状态 [px, py, pz, vx, vy, vz, qw, qx, qy, qz]
        """
        if self.trajectory_reference is None:
            # 如果没有轨迹，返回悬停状态
            return np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        
        # 计算从轨迹开始到现在经过的时间
        rel_time = t - self.trajectory_start_time
        
        # 获取轨迹总时长
        total_duration = self.trajectory_reference[-1].time
        
        # 循环轨迹：使用模运算使轨迹循环
        rel_time = rel_time % total_duration if total_duration > 0 else 0.0
        
        # 在轨迹中找到最近的点
        if rel_time <= 0:
            # 时间在轨迹开始之前，返回第一个点
            point = self.trajectory_reference[0]
        elif rel_time >= self.trajectory_reference[-1].time:
            # 时间在轨迹结束之后，循环回到开始（不应该到这里，因为有模运算）
            point = self.trajectory_reference[0]
        else:
            # 线性插值
            point = self._interpolate_trajectory(rel_time)
        
        # 从偏航角转换为四元数
        yaw = point.yaw
        quat = R.from_euler('z', yaw).as_quat()  # [x, y, z, w]
        # 转换为我们的格式 [w, x, y, z]
        quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])
        
        # 构建参考状态
        reference_state = np.array([
            point.position.x, point.position.y, point.position.z,
            point.velocity.x, point.velocity.y, point.velocity.z,
            quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]
        ])
        
        return reference_state
    
    def _interpolate_trajectory(self, t):
        """
        在轨迹中线性插值
        """
        # 找到t所在的区间
        for i in range(len(self.trajectory_reference) - 1):
            if (self.trajectory_reference[i].time <= t <= 
                self.trajectory_reference[i + 1].time):
                
                p1 = self.trajectory_reference[i]
                p2 = self.trajectory_reference[i + 1]
                
                # 插值因子
                dt = p2.time - p1.time
                if dt == 0:
                    alpha = 0
                else:
                    alpha = (t - p1.time) / dt
                
                # 创建插值点
                class InterpolatedPoint:
                    def __init__(self):
                        self.time = t
                        self.position = type('Position', (), {})()
                        self.velocity = type('Velocity', (), {})()
                        self.acceleration = type('Acceleration', (), {})()
                
                point = InterpolatedPoint()
                
                # 插值位置
                point.position.x = p1.position.x + alpha * (p2.position.x - p1.position.x)
                point.position.y = p1.position.y + alpha * (p2.position.y - p1.position.y)
                point.position.z = p1.position.z + alpha * (p2.position.z - p1.position.z)
                
                # 插值速度
                point.velocity.x = p1.velocity.x + alpha * (p2.velocity.x - p1.velocity.x)
                point.velocity.y = p1.velocity.y + alpha * (p2.velocity.y - p1.velocity.y)
                point.velocity.z = p1.velocity.z + alpha * (p2.velocity.z - p1.velocity.z)
                
                # 插值加速度
                point.acceleration.x = p1.acceleration.x + alpha * (p2.acceleration.x - p1.acceleration.x)
                point.acceleration.y = p1.acceleration.y + alpha * (p2.acceleration.y - p1.acceleration.y)
                point.acceleration.z = p1.acceleration.z + alpha * (p2.acceleration.z - p1.acceleration.z)
                
                # 插值偏航角
                point.yaw = p1.yaw + alpha * (p2.yaw - p1.yaw)
                point.yaw_rate = p1.yaw_rate + alpha * (p2.yaw_rate - p1.yaw_rate)
                
                return point
        
        # 如果没找到区间，返回最后一个点
        return self.trajectory_reference[-1]
    
    def solve_trajectory(self, x0, current_time, verbose=False):
        """
        求解轨迹跟踪MPC
        
        Args:
            x0: 当前状态
            current_time: 当前时间
            verbose: 是否打印详细信息
            
        Returns:
            simU: 控制输入序列
            simX: 状态预测序列
        """
        ocp_solver = self.ocp_solver
        
        # 设置初始状态约束
        ocp_solver.set(0, "lbx", x0)
        ocp_solver.set(0, "ubx", x0)
        
        # 设置轨迹参考
        dt = self.Tf / self.N
        for i in range(self.N + 1):
            future_time = current_time + i * dt
            reference_state = self.get_reference_at_time(future_time)
            
            if i < self.N:
                # 设置阶段成本的参考
                yref = np.concatenate([reference_state, np.zeros(4)])  # 状态+控制参考
                ocp_solver.set(i, "yref", yref)
            else:
                # 设置终端成本的参考
                ocp_solver.set(i, "yref", reference_state)
        
        # 求解
        status = ocp_solver.solve()
        
        if verbose:
            self.ocp_solver.print_statistics()
        
        if status != 0:
            raise Exception(f'acados returned status {status}.')
        
        N = self.N
        nx = self.model.get_acados_model().x.size()[0]
        nu = self.model.get_acados_model().u.size()[0]
        
        simX = np.ndarray((N+1, nx))
        simU = np.ndarray((N, nu))
        
        # 获取解
        for i in range(N):
            simX[i,:] = self.ocp_solver.get(i, "x")
            simU[i,:] = self.ocp_solver.get(i, "u")
        simX[N,:] = self.ocp_solver.get(N, "x")
        
        return simU, simX