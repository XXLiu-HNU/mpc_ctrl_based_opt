#!/usr/bin/env python3
import rospy
import numpy as np
import os
import threading
from collections import defaultdict
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from mpc_ctrl.msg import PositionCommand
from mavros_msgs.msg import AttitudeTarget  # 新增AttitudeTarget消息类型
from message_filters import ApproximateTimeSynchronizer, Subscriber
from tf.transformations import euler_from_quaternion

def normalize_yaw(yaw):
    """将角度归一化到[-π, π]范围"""
    yaw = yaw % (2 * np.pi)
    return yaw - 2 * np.pi if yaw > np.pi else yaw

class EnhancedPIDDataRecorder:
    def __init__(self):
        rospy.init_node('enhanced_pid_data_recorder')
        
        # 初始化参数和路径
        save_path_name = rospy.get_param("controller/save_path", "~/experiment/mpc_ctrl_ws/src/mpc_ctrl/scripts/mpc_tracking_data.csv")
        self.odom_topic = rospy.get_param("controller/odom_topic", "/mavros/local_position/odom")
        self.cmd_topic = rospy.get_param("controller/target_topic", "/planning/pos_cmd")
        self.imu_topic = rospy.get_param("controller/imu_topic", "/mavros/imu/data")
        self.attitude_target_topic = rospy.get_param("controller/attitude_target_topic", "/mavros/setpoint_raw/attitude")  # 新增目标姿态话题
        # 创建保存目录
        self.save_path = os.path.expanduser(save_path_name)
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        rospy.loginfo(f"数据将保存至: {self.save_path}")

        # 初始化数据结构（新增油门值存储）
        self.data = defaultdict(list)
        self.data_lock = threading.Lock()
        self.prev_vel = np.zeros(3)  # 上一时刻线速度
        self.prev_time = None  # 上一时刻时间戳
        self.saving_flag = False
        self.data_count = 0
        self.prev_yaw = 0.0
        # 新增：缓存上一时刻的目标角速度和油门（防止消息丢失）
        self.prev_target_ang_vel = np.zeros(3)
        self.prev_throttle = 0.0

        # 初始化订阅器（包含姿态目标话题）
        self._init_subscribers()
        
        # 注册关闭钩子
        rospy.on_shutdown(self.shutdown_hook)

    def _init_subscribers(self):
        """初始化消息订阅器（同步odom、cmd、imu、attitude_target）"""
        rospy.loginfo(f"订阅话题: \n- 里程计: {self.odom_topic}\n- 控制指令: {self.cmd_topic}\n- IMU: {self.imu_topic}\n- 目标姿态: {self.attitude_target_topic}")
        
        try:
            odom_sub = Subscriber(self.odom_topic, Odometry)
            target_sub = Subscriber(self.cmd_topic, PositionCommand)
            imu_sub = Subscriber(self.imu_topic, Imu)
            attitude_target_sub = Subscriber(self.attitude_target_topic, AttitudeTarget)  # 新增目标姿态订阅
            
            # 同步四个话题，时间偏差允许0.05秒
            self.ts = ApproximateTimeSynchronizer(
                [odom_sub, target_sub, imu_sub, attitude_target_sub],
                queue_size=50,  # 增大队列适应多话题同步
                slop=0.05
            )
            self.ts.registerCallback(self.sync_callback)  # 回调接收四个消息
        except Exception as e:
            rospy.logerr(f"订阅器初始化失败: {str(e)}")
            raise

    def sync_callback(self, odom_msg, target_msg, imu_msg, attitude_target_msg):  # 新增attitude_target_msg参数
        """同步回调：处理四话题数据，从AttitudeTarget获取目标角速度和油门"""
        if self.saving_flag:
            return
            
        try:
            current_time = odom_msg.header.stamp.to_sec()  # 以odom时间戳为基准
            
            with self.data_lock:
                # 记录基础位置和Yaw
                self._record_basic_data(current_time, odom_msg, target_msg)
                
                # 记录线速度和实际角速度
                current_vel = self._record_actual_velocity(odom_msg)
                self._record_actual_angular_velocity(imu_msg)
                
                # 计算实际加速度
                self._calculate_actual_acceleration(current_time, current_vel)
                
                # 记录目标线速度和线加速度（来自PositionCommand）
                self._record_target_linear_dynamics(target_msg)
                
                # 新增：从AttitudeTarget获取目标角速度和油门
                self._record_target_angular_and_throttle(attitude_target_msg)
                
                self.data_count += 1
                
        except Exception as e:
            rospy.logerr(f"回调处理异常: {str(e)}")
            self._emergency_save()

    def _record_basic_data(self, t, odom, target):
        """记录位置和Yaw角"""
        self.data['timestamp'].append(t)
        # 实际位置
        self.data['actual_x'].append(odom.pose.pose.position.x)
        self.data['actual_y'].append(odom.pose.pose.position.y)
        self.data['actual_z'].append(odom.pose.pose.position.z)
        # 目标位置
        self.data['target_x'].append(target.position.x)
        self.data['target_y'].append(target.position.y)
        self.data['target_z'].append(target.position.z)
        # Yaw角
        self._record_yaw_data(odom, target)

    def _record_yaw_data(self, odom, target):
        """提取Yaw角"""
        orientation = odom.pose.pose.orientation
        try:
            (_, _, current_yaw) = euler_from_quaternion([
                orientation.x, orientation.y, orientation.z, orientation.w
            ])
        except:
            current_yaw = self.prev_yaw
        self.data['target_yaw'].append(target.yaw)
        self.data['actual_yaw'].append(normalize_yaw(current_yaw))
        self.prev_yaw = current_yaw

    def _record_actual_velocity(self, odom):
        """记录实际线速度"""
        vel_x = odom.twist.twist.linear.x
        vel_y = odom.twist.twist.linear.y
        vel_z = odom.twist.twist.linear.z
        
        self.data['actual_vel_x'].append(vel_x)
        self.data['actual_vel_y'].append(vel_y)
        self.data['actual_vel_z'].append(vel_z)
        
        return np.array([vel_x, vel_y, vel_z])

    def _calculate_actual_acceleration(self, current_time, current_vel):
        """计算实际加速度（速度微分）"""
        try:
            if self.prev_time is None:
                self.data['actual_acc_x'].append(0.0)
                self.data['actual_acc_y'].append(0.0)
                self.data['actual_acc_z'].append(0.0)
                self.prev_time = current_time
                self.prev_vel = current_vel
                return

            dt = current_time - self.prev_time
            if dt <= 0:
                last_acc_x = self.data['actual_acc_x'][-1] if self.data['actual_acc_x'] else 0.0
                last_acc_y = self.data['actual_acc_y'][-1] if self.data['actual_acc_y'] else 0.0
                last_acc_z = self.data['actual_acc_z'][-1] if self.data['actual_acc_z'] else 0.0
                self.data['actual_acc_x'].append(last_acc_x)
                self.data['actual_acc_y'].append(last_acc_y)
                self.data['actual_acc_z'].append(last_acc_z)
                return

            acc = (current_vel - self.prev_vel) / dt
            self.data['actual_acc_x'].append(acc[0])
            self.data['actual_acc_y'].append(acc[1])
            self.data['actual_acc_z'].append(acc[2])
            
            self.prev_time = current_time
            self.prev_vel = current_vel
            
        except Exception as e:
            rospy.logwarn(f"加速度计算异常: {str(e)}，使用0填充")
            self.data['actual_acc_x'].append(0.0)
            self.data['actual_acc_y'].append(0.0)
            self.data['actual_acc_z'].append(0.0)

    def _record_actual_angular_velocity(self, imu):
        """记录实际角速度（来自IMU）"""
        self.data['actual_ang_vel_x'].append(imu.angular_velocity.x)
        self.data['actual_ang_vel_y'].append(imu.angular_velocity.y)
        self.data['actual_ang_vel_z'].append(imu.angular_velocity.z)

    def _record_target_linear_dynamics(self, target):
        """记录目标线速度和线加速度（来自PositionCommand）"""
        # 目标线速度
        self.data['target_vel_x'].append(target.velocity.x)
        self.data['target_vel_y'].append(target.velocity.y)
        self.data['target_vel_z'].append(target.velocity.z)
        # 目标线加速度
        self.data['target_acc_x'].append(target.acceleration.x)
        self.data['target_acc_y'].append(target.acceleration.y)
        self.data['target_acc_z'].append(target.acceleration.z)

    def _record_target_angular_and_throttle(self, attitude_target):
        """新增：从AttitudeTarget记录目标角速度和油门值"""
        try:
            # 目标角速度（body_rate为机体坐标系下的目标角速度）
            target_ang_vel_x = attitude_target.body_rate.x
            target_ang_vel_y = attitude_target.body_rate.y
            target_ang_vel_z = attitude_target.body_rate.z
            # 油门值（throttle范围通常为0~1）
            throttle = attitude_target.thrust
            
            # 记录目标角速度（保持原有键名，确保数据顺序不变）
            self.data['target_ang_vel_x'].append(target_ang_vel_x)
            self.data['target_ang_vel_y'].append(target_ang_vel_y)
            self.data['target_ang_vel_z'].append(target_ang_vel_z)
            # 新增：记录油门值
            self.data['target_throttle'].append(throttle)
            
            # 缓存当前值，用于消息丢失时填充
            self.prev_target_ang_vel = np.array([target_ang_vel_x, target_ang_vel_y, target_ang_vel_z])
            self.prev_throttle = throttle
            
        except Exception as e:
            rospy.logwarn(f"目标角速度/油门提取异常: {str(e)}，使用上一时刻值填充")
            # 异常时使用缓存值填充
            self.data['target_ang_vel_x'].append(self.prev_target_ang_vel[0])
            self.data['target_ang_vel_y'].append(self.prev_target_ang_vel[1])
            self.data['target_ang_vel_z'].append(self.prev_target_ang_vel[2])
            self.data['target_throttle'].append(self.prev_throttle)

    def shutdown_hook(self):
        """关闭时保存数据"""
        if not self.saving_flag and self.data_count > 0:
            rospy.logwarn("接收到关闭信号，正在保存数据...")
            self.save_data()

    def _emergency_save(self):
        """紧急保存"""
        rospy.logerr("触发紧急保存机制")
        self.save_data()

    def save_data(self):
        """保存数据到CSV（新增油门列，保持原有顺序）"""
        if self.saving_flag or self.data_count == 0:
            return
            
        self.saving_flag = True
        rospy.loginfo("开始数据保存流程...")
        
        try:
            with self.data_lock:
                base_length = len(self.data['timestamp'])
                if base_length == 0:
                    rospy.logwarn("无有效数据需要保存")
                    return
                
                # 对齐所有数据长度
                min_length = min([len(self.data[key]) for key in self.data.keys()] + [base_length])
                
                # 数据列定义（原有顺序不变，末尾新增油门列）
                columns = [
                    # 时间戳
                    'timestamp',
                    # 位置
                    'actual_x', 'target_x',
                    'actual_y', 'target_y',
                    'actual_z', 'target_z',
                    # Yaw角
                    'actual_yaw', 'target_yaw',
                    # 线速度
                    'actual_vel_x', 'target_vel_x',
                    'actual_vel_y', 'target_vel_y',
                    'actual_vel_z', 'target_vel_z',
                    # 线加速度
                    'actual_acc_x', 'target_acc_x',
                    'actual_acc_y', 'target_acc_y',
                    'actual_acc_z', 'target_acc_z',
                    # 角速度（目标角速度 now from AttitudeTarget）
                    'actual_ang_vel_x', 'target_ang_vel_x',
                    'actual_ang_vel_y', 'target_ang_vel_y',
                    'actual_ang_vel_z', 'target_ang_vel_z',
                    # 新增：油门值（添加到末尾，不影响原有顺序）
                    'target_throttle'
                ]
                
                # 准备数据
                data_to_save = []
                for col in columns:
                    if len(self.data[col]) < min_length:
                        rospy.logwarn(f"字段 {col} 数据长度不足，自动截断")
                    data_to_save.append(self.data[col][:min_length])
                
                # 保存为CSV
                data_array = np.column_stack(data_to_save)
                header = ",".join(columns)
                np.savetxt(self.save_path, data_array, header=header, delimiter=',', fmt='%.6f')
                rospy.loginfo(f"成功保存 {min_length} 条数据 (总接收 {self.data_count} 条)")
                
        except Exception as e:
            rospy.logerr(f"保存过程中发生错误: {str(e)}")
        finally:
            self.saving_flag = False

    def run(self):
        """主循环"""
        rospy.Timer(rospy.Duration(60), lambda _: self.save_data(), oneshot=False)  # 定时保存
        rospy.spin()

if __name__ == '__main__':
    recorder = None
    try:
        recorder = EnhancedPIDDataRecorder()
        recorder.run()
    except rospy.ROSInterruptException:
        if recorder:
            recorder.save_data()
    except Exception as e:
        rospy.logerr(f"程序异常终止: {str(e)}")
        if recorder:
            recorder._emergency_save()
    finally:
        if recorder:
            recorder.save_data()