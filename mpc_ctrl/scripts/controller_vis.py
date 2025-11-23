#!/usr/bin/env python3
"""
PID/MPC控制器性能可视化分析工具
功能：读取跟踪数据CSV和控制器日志文件，生成包含位置、速度、加速度、角速度、偏航角、油门值和推力值的可视化图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def load_data(file_path):
    """加载并预处理MPC跟踪数据"""
    columns = [
        'timestamp',         # 时间戳
        'actual_x',          # X轴实际位置
        'target_x',          # X轴目标位置
        'actual_y',          # Y轴实际位置
        'target_y',          # Y轴目标位置
        'actual_z',          # Z轴实际位置
        'target_z',          # Z轴目标位置
        'actual_yaw',        # 实际偏航角
        'target_yaw',        # 目标偏航角
        'actual_vel_x',      # X轴实际速度
        'target_vel_x',      # X轴目标速度
        'actual_vel_y',      # Y轴实际速度
        'target_vel_y',      # Y轴目标速度
        'actual_vel_z',      # Z轴实际速度
        'target_vel_z',      # Z轴目标速度
        'actual_acc_x',      # X轴实际加速度
        'target_acc_x',      # X轴目标加速度
        'actual_acc_y',      # Y轴实际加速度
        'target_acc_y',      # Y轴目标加速度
        'actual_acc_z',      # Z轴实际加速度
        'target_acc_z',      # Z轴目标加速度
        'actual_ang_vel_x',  # X轴实际角速度
        'target_ang_vel_x',  # X轴目标角速度
        'actual_ang_vel_y',  # Y轴实际角速度
        'target_ang_vel_y',  # Y轴目标角速度
        'actual_ang_vel_z',  # Z轴实际角速度
        'target_ang_vel_z',  # Z轴目标角速度
        'target_throttle'    # 目标油门值
    ]
    
    data = pd.read_csv(file_path, skiprows=1, names=columns)
    
    # 时间标准化处理（从0开始）
    data['time'] = data['timestamp'] - data['timestamp'].min()
    
    # 角度归一化处理
    data['actual_yaw'] = np.arctan2(np.sin(data['actual_yaw']), np.cos(data['actual_yaw']))
    data['target_yaw'] = np.arctan2(np.sin(data['target_yaw']), np.cos(data['target_yaw']))
    
    return data

def load_thrust_data(file_path):
    """加载控制器日志中的推力数据"""
    # 定义列名（对应日志格式）
    columns = [
        'timestamp',     # 时间戳(秒)
        'iterations',    # 迭代次数
        'initialError',  # 初始误差
        'finalError',    # 最终误差
        'converged',     # 是否收敛
        'opt_cost',      # 优化成本
        'all_cost',      # 总成本
        'thrust'         # 推力值(input[0])
    ]
    
    # 读取日志文件
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = list(map(float, line.split()))
            if len(parts) == 8:  # 确保数据格式正确
                data.append(parts)
    
    # 转换为DataFrame
    df = pd.DataFrame(data, columns=columns)
    
    # 时间标准化（与主数据保持一致的时间基准格式）
    if not df.empty:
        df['time'] = df['timestamp'] - df['timestamp'].min()
    
    return df

def plot_performance(data, thrust_data):
    """生成性能分析图表，包含推力值可视化"""
    # 创建5行3列的子图布局
    fig = plt.figure(figsize=(18, 36))
    fig.suptitle('Comprehensive Analysis of MPC Control Performance', fontsize=20, y=0.95)
    
    # 定义子图布局参数和颜色配置
    axis_config = {
        'x': {'row': 0, 'label': 'X-Axis'},
        'y': {'row': 1, 'label': 'Y-Axis'},
        'z': {'row': 2, 'label': 'Z-Axis'},
        'yaw': {'row': 3, 'label': 'Yaw'},
        'ang_vel': {'row': 4, 'label': 'Angular Velocity'}
    }
    colors = {
        'positive_error': 'lightcoral',    # 实际值 > 目标值的误差区域
        'negative_error': 'lightgreen',    # 实际值 < 目标值的误差区域
    }

    
    # 遍历各轴生成图表
    for axis in ['x', 'y', 'z', 'yaw', 'ang_vel']:
        row = axis_config[axis]['row']
        
        if axis in ['x', 'y', 'z']:
            # 位置跟踪图表
            plt.subplot(5, 3, row*3 + 1)
            plot_with_error_fill(
                data, f'actual_{axis}', f'target_{axis}', 
                f"{axis_config[axis]['label']} Position Tracking",
                'Position (m)', colors
            )
            
            # 速度跟踪图表
            plt.subplot(5, 3, row*3 + 2)
            plot_with_error_fill(
                data, f'actual_vel_{axis}', f'target_vel_{axis}', 
                f"{axis_config[axis]['label']} Velocity Tracking",
                'Velocity (m/s)', colors
            )
            
            # 加速度跟踪图表
            plt.subplot(5, 3, row*3 + 3)
            plot_with_error_fill(
                data, f'actual_acc_{axis}', f'target_acc_{axis}', 
                f"{axis_config[axis]['label']} Acceleration Tracking",
                'Acceleration (m/s²)', colors
            )
        elif axis == 'yaw':
            # 偏航角跟踪图表
            plt.subplot(5, 3, row*3 + 1)
            plot_with_error_fill(
                data, 'actual_yaw', 'target_yaw', 
                "Yaw Angle Tracking Performance",
                'Angle (rad)', colors
            )
            
            # 替换为推力值可视化（原Yaw Rate Tracking位置）
            plt.subplot(5, 3, row*3 + 2)
            plot_thrust(thrust_data)
            
            # 油门值可视化（保持不变）
            plt.subplot(5, 3, row*3 + 3)
            plot_throttle(data)
        elif axis == 'ang_vel':
            # 滚转角速度跟踪
            plt.subplot(5, 3, row*3 + 1)
            plot_with_error_fill(
                data, 'actual_ang_vel_x', 'target_ang_vel_x', 
                "Roll Angular Velocity Tracking",
                'Angular Velocity (rad/s)', colors
            )
            
            # 俯仰角速度跟踪
            plt.subplot(5, 3, row*3 + 2)
            plot_with_error_fill(
                data, 'actual_ang_vel_y', 'target_ang_vel_y', 
                "Pitch Angular Velocity Tracking",
                'Angular Velocity (rad/s)', colors
            )
            
            # 偏航角速度跟踪
            plt.subplot(5, 3, row*3 + 3)
            plot_with_error_fill(
                data, 'actual_ang_vel_z', 'target_ang_vel_z', 
                "Yaw Angular Velocity Tracking",
                'Angular Velocity (rad/s)', colors
            )
    
    plt.tight_layout(pad=3.0)
    plt.savefig('full_analysis_with_thrust.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_with_error_fill(data, actual_col, target_col, title, ylabel, colors):
    """通用绘图函数：绘制实际值和目标值曲线，并填充差值区域"""
    plt.plot(data['time'], data[actual_col], 'b-', label='Actual', linewidth=1.2)
    plt.plot(data['time'], data[target_col], 'r--', label='Target', linewidth=1.2)
    
    # 填充差值区域
    plt.fill_between(
        data['time'], 
        data[actual_col], 
        data[target_col], 
        where=(data[actual_col] > data[target_col]),
        interpolate=True,
        color=colors['positive_error'], 
        alpha=0.4,
        label='Positive Error'
    )
    plt.fill_between(
        data['time'], 
        data[actual_col], 
        data[target_col], 
        where=(data[actual_col] <= data[target_col]),
        interpolate=True,
        color=colors['negative_error'], 
        alpha=0.4,
        label='Negative Error'
    )
    
    # 添加网格、标题和标签
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='upper right')

def plot_throttle(data):
    """油门值可视化"""
    plt.plot(data['time'], data['target_throttle'], 'g-', label='Target Throttle', linewidth=1.5)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axhline(y=1, color='k', linestyle='--', alpha=0.3, label='Max Throttle')
    plt.title("Target Throttle Profile")
    plt.ylabel('Throttle Value')
    plt.ylim(-0.1, 1.1)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='upper right')

def plot_thrust(thrust_data):
    """推力值可视化（新增函数）"""
    if thrust_data.empty:
        plt.title("Thrust Data (No Data Available)")
        return
    
    # 绘制推力值曲线
    plt.plot(thrust_data['time'], thrust_data['thrust'], 'purple', linewidth=1.5, label='Calculated Thrust')
    
    # 添加统计参考线
    mean_thrust = thrust_data['thrust'].mean()
    max_thrust = thrust_data['thrust'].max()
    min_thrust = thrust_data['thrust'].min()
    
    plt.axhline(y=mean_thrust, color='orange', linestyle='--', alpha=0.7, label=f'Mean: {mean_thrust:.4f}')
    plt.axhline(y=max_thrust, color='red', linestyle=':', alpha=0.5, label=f'Max: {max_thrust:.4f}')
    plt.axhline(y=min_thrust, color='blue', linestyle=':', alpha=0.5, label=f'Min: {min_thrust:.4f}')
    
    # 图表配置
    plt.title("MPC Calculated Thrust Profile")
    plt.ylabel('Thrust Value')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='upper right')

def print_metrics(data, thrust_data):
    """打印性能指标，包含推力值分析"""
    print("\n=== Comprehensive Performance Metrics ===")
    
    axis_config = {
        'x': {'label': 'X-Axis'},
        'y': {'label': 'Y-Axis'},
        'z': {'label': 'Z-Axis'},
        'yaw': {'label': 'Yaw'}
    }
    
    # 位置、速度、加速度性能指标分析
    for axis in ['x', 'y', 'z']:
        print(f"\n{axis_config[axis]['label']} Performance:")
        
        # 位置误差分析
        pos_error = data[f'actual_{axis}'] - data[f'target_{axis}']
        print("[Position]")
        print(f"  Max Positive Error: {pos_error.max():.4f} m")
        print(f"  Max Negative Error: {pos_error.min():.4f} m")
        print(f"  Steady-state Error: {pos_error.iloc[-100:].mean():.4f} m")
        print(f"  RMS Error: {np.sqrt((pos_error**2).mean()):.4f} m")

        # 速度误差分析
        vel_error = data[f'actual_vel_{axis}'] - data[f'target_vel_{axis}']
        print("\n[Velocity]")
        print(f"  Max Positive Error: {vel_error.max():.4f} m/s")
        print(f"  Max Negative Error: {vel_error.min():.4f} m/s")
        print(f"  RMS Error: {np.sqrt((vel_error**2).mean()):.4f} m/s")

        # 加速度误差分析
        acc_error = data[f'actual_acc_{axis}'] - data[f'target_acc_{axis}']
        print("\n[Acceleration]")
        print(f"  Max Positive Error: {acc_error.max():.4f} m/s²")
        print(f"  Max Negative Error: {acc_error.min():.4f} m/s²")
        print(f"  Mean Absolute Error: {acc_error.abs().mean():.4f} m/s²")
    
    # 角速度性能指标分析
    print("\n=== Angular Velocity Performance ===")
    angle_names = {'x': 'Roll', 'y': 'Pitch', 'z': 'Yaw'}
    for axis in ['x', 'y', 'z']:
        ang_vel_error = data[f'actual_ang_vel_{axis}'] - data[f'target_ang_vel_{axis}']
        print(f"\n[{angle_names[axis]}]")
        print(f"  Max Positive Error: {ang_vel_error.max():.4f} rad/s")
        print(f"  Max Negative Error: {ang_vel_error.min():.4f} rad/s")
        print(f"  RMS Error: {np.sqrt((ang_vel_error**2).mean()):.4f} rad/s")
        print(f"  Mean Absolute Error: {ang_vel_error.abs().mean():.4f} rad/s")

    # 偏航角误差分析
    yaw_error = data['actual_yaw'] - data['target_yaw']
    print("\n=== Yaw Performance ===")
    print(f"  Max Positive Error: {yaw_error.max():.4f} rad")
    print(f"  Max Negative Error: {yaw_error.min():.4f} rad")
    print(f"  Mean Absolute Error: {yaw_error.abs().mean():.4f} rad")
    print(f"  RMS Error: {np.sqrt((yaw_error**2).mean()):.4f} rad")

    # 油门值性能指标
    throttle = data['target_throttle']
    print("\n=== Throttle Performance ===")
    print(f"  Max Value: {throttle.max():.4f}")
    print(f"  Min Value: {throttle.min():.4f}")
    print(f"  Mean Value: {throttle.mean():.4f}")
    print(f"  Std Deviation: {throttle.std():.4f}")
    print(f"  Max Fluctuation: {throttle.max() - throttle.min():.4f}")

    # 推力值性能指标（新增）
    if not thrust_data.empty:
        print("\n=== Thrust Performance ===")
        print(f"  Max Thrust: {thrust_data['thrust'].max():.4f}")
        print(f"  Min Thrust: {thrust_data['thrust'].min():.4f}")
        print(f"  Mean Thrust: {thrust_data['thrust'].mean():.4f}")
        print(f"  Std Deviation: {thrust_data['thrust'].std():.4f}")
        print(f"  Max Fluctuation: {thrust_data['thrust'].max() - thrust_data['thrust'].min():.4f}")
    else:
        print("\n=== Thrust Performance ===")
        print("  No thrust data available for analysis")

if __name__ == '__main__':
    # 获取当前脚本所在的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 构建数据文件路径
    tracking_file = os.path.join(script_dir, "mpc_tracking_data.csv")  
    thrust_file = os.path.join(script_dir, "controller_log.txt")  
    
    # 加载数据
    tracking_data = load_data(tracking_file)
    thrust_data = load_thrust_data(thrust_file)
    
    # 生成可视化图表
    plot_performance(tracking_data, thrust_data)
    
    # 输出性能指标
    print_metrics(tracking_data, thrust_data)