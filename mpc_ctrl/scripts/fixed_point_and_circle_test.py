import rospy
import numpy as np
from geometry_msgs.msg import Point, Vector3, PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Header
from mpc_ctrl.msg import PositionCommand
import nav_msgs.msg
from mavros_msgs.msg import State
from mavros_msgs.srv import SetMode, CommandBool, CommandBoolRequest, SetModeRequest
import tf

def normalize_yaw(yaw):
    yaw = yaw % (2 * np.pi)
    return yaw - 2 * np.pi if yaw > np.pi else yaw

class CircleTrajectoryPublisher:
    def __init__(self):
        rospy.init_node('circle_trajectory_publisher')
        rospy.loginfo("初始化阶段")
        # 参数初始化
        self.traj_mode = rospy.get_param('controller/traj_mode', 0)  # 轨迹模式选择
        self.real_world_flag = rospy.get_param('controller/real_world_flag', False)  # 是否在真实环境中运行
        self.radius = rospy.get_param('controller/circle_radius', 3.0)
        self.time_step = rospy.get_param('controller/time_to_max_radius', 30.0)
        self.yaw_control = rospy.get_param('controller/yaw_control', True)
        self.velocity = rospy.get_param('controller/velocity', 2.5)
        self.takeoff_height = rospy.get_param('controller/takeoff_height', 1.0)
        self.max_path_length = rospy.get_param('controller/max_path_length', 600)
        self.z_amplitude = rospy.get_param('controller/z_amplitude', 0.2)
        self.z_frequency = rospy.get_param('controller/z_frequency', 0.2)
        self.takeoff_threshold = 0.05  # 起飞位置误差阈值
        
        # 话题与服务（适配iris_0命名空间）
        cmd_topic = rospy.get_param("controller/target_topic", "/planning/pos_cmd")
        self.pub_cmd = rospy.Publisher(cmd_topic, PositionCommand, queue_size=10)
        self.pub_ref = rospy.Publisher('/reference_trajectory', Path, queue_size=10)
        self.pub_setpoint = rospy.Publisher('/mavros/setpoint_position/local', PoseStamped, queue_size=10)
        
        # MAVROS服务（等待服务可用后创建客户端）
        rospy.wait_for_service('/mavros/cmd/arming')
        self.arm_client = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
        rospy.wait_for_service('/mavros/set_mode')
        self.set_mode_client = rospy.ServiceProxy('/mavros/set_mode', SetMode)
        
        # 订阅状态和里程计
        self.sub_state = rospy.Subscriber('/mavros/state', State, self.state_callback)
        odom_topic = rospy.get_param("controller/odom_topic", "/mavros/local_position/odom")
        self.sub_odom = rospy.Subscriber(odom_topic, nav_msgs.msg.Odometry, self.odom_callback)

        self.current_state = State()
        self.curr_pos = Point(0, 0, 0)
        # 状态变量
        self.armed = False
        self.offboard = False
        self.takeoff_complete = False
        self.state = 0  # 0-初始化前准备，1-初始化（模式切换），2-起飞，3-圆形轨迹
        self.last_req = rospy.Time.now()  # 服务请求时间戳
        self.start_time = rospy.Time.now()
        self.curr_yaw = 0.0 
        self.first_init = True
        
        # 轨迹参数
        self.rate = 100
        self.dt = 1.0 / self.rate
        self.theta = 0.0
        self.current_radius = 0.0
        self.current_velocity = 0.0
        self.step = self.time_step * self.rate
        self.z_phase = 0.0
        
        # 路径可视化
        self.reference_path = Path()
        self.reference_path.header.frame_id = 'world'

        # 轨迹规划参数（可根据无人机性能调整）
        self.takeoff_params = {
            "total_time": 5.0,       # 总时间缩短（加速+巡航+减速）
            "accel_time": 2.0,       # 加速时间不变（确保加速到max_vel）
            "cruise_time": 0.5,      # 大幅缩短巡航时间（仅作为过渡）
            "decel_time": 2.0,       # 减速时间不变（确保速度从max_vel减到0）
            "max_vel": 0.5,          # 最大速度不变
            "max_accel": 0.25,       # 加速度不变
            "stable_threshold": 0.05 # 稳定阈值不变
        }
        # 轨迹状态变量
        self.takeoff_start_time = None  # 起飞开始时间
        self.trajectory_z = 0.0  # 当前目标Z位置
        self.trajectory_vel_z = 0.0  # 当前目标Z速度
        self.trajectory_acc_z = 0.0  # 当前目标Z加速度

        # 提前发送100个初始指令（满足PX4 OFFBOARD模式要求）
        self.send_initial_setpoints()
        
        # 启动定时器
        self.timer = rospy.Timer(rospy.Duration(self.dt), self.timer_callback)

    def update_takeoff_trajectory(self):
        """实时生成平滑的Z轴起飞轨迹（位置、速度、加速度）"""
        if self.takeoff_start_time is None:
            # 记录起飞开始时间和初始位置
            self.takeoff_start_time = rospy.Time.now()
            self.initial_z = self.curr_pos.z if self.curr_pos else 0.0
            rospy.loginfo(f"开始平滑起飞，初始高度: {self.initial_z:.2f}m，目标高度: {self.takeoff_height:.2f}m")
            return

        # 计算当前起飞时长（相对于起飞开始的时间）
        t = (rospy.Time.now() - self.takeoff_start_time).to_sec()
        total_t = self.takeoff_params["total_time"]
        accel_t = self.takeoff_params["accel_time"]
        cruise_t = self.takeoff_params["cruise_time"]
        decel_t = self.takeoff_params["decel_time"]

        # 1. 轨迹终点约束（目标高度）
        target_z = self.takeoff_height
        initial_z = self.initial_z
        delta_z = target_z - initial_z  # 总上升高度

        # 2. 分段轨迹规划（确保位置、速度、加速度连续）
        if t <= 0:
            # 起飞前初始状态
            self.trajectory_z = initial_z
            self.trajectory_vel_z = 0.0
            self.trajectory_acc_z = 0.0

        # 阶段1：加速上升（0 ~ accel_t）
        elif t <= accel_t:
            # 加速度恒定为max_accel，速度线性增加
            self.trajectory_acc_z = self.takeoff_params["max_accel"]
            self.trajectory_vel_z = self.trajectory_acc_z * t  # v = a*t
            # 限制加速阶段最大速度不超过max_vel（安全冗余）
            self.trajectory_vel_z = min(self.trajectory_vel_z, self.takeoff_params["max_vel"])
            # 位置：s = 0.5*a*t²（从初始位置开始）
            self.trajectory_z = initial_z + 0.5 * self.trajectory_acc_z * t**2
            rospy.logdebug_throttle(1, f"加速阶段: t={t:.1f}s, z={self.trajectory_z:.2f}m, vel={self.trajectory_vel_z:.2f}m/s")

        # 阶段2：匀速巡航（accel_t ~ accel_t+cruise_t）
        elif t <= accel_t + cruise_t:
            # 加速度为0，速度保持最大速度
            self.trajectory_acc_z = 0.0
            self.trajectory_vel_z = self.takeoff_params["max_vel"]
            # 位置：加速阶段终点位置 + 匀速阶段位移
            z_accel_end = initial_z + 0.5 * self.takeoff_params["max_accel"] * accel_t**2
            t_cruise = t - accel_t  # 巡航阶段已过时间
            self.trajectory_z = z_accel_end + self.trajectory_vel_z * t_cruise
            rospy.logdebug_throttle(1, f"巡航阶段: t={t:.1f}s, z={self.trajectory_z:.2f}m, vel={self.trajectory_vel_z:.2f}m/s")

        # 阶段3：减速上升（accel_t+cruise_t ~ total_t）
        elif t <= total_t:
            # 加速度为负（减速），速度线性减小
            self.trajectory_acc_z = -self.takeoff_params["max_accel"]
            t_decel = t - (accel_t + cruise_t)  # 减速阶段已过时间
            # 计算当前速度（核心：增加非负约束，确保速度≥0）
            self.trajectory_vel_z = max(
                self.takeoff_params["max_vel"] + self.trajectory_acc_z * t_decel,
                0.0  # 速度下限为0，避免出现负值
            )
            # 位置：巡航阶段终点位置 + 减速阶段位移（匀减速运动）
            z_cruise_end = initial_z + 0.5 * self.takeoff_params["max_accel"] * accel_t**2 + self.takeoff_params["max_vel"] * cruise_t
            self.trajectory_z = z_cruise_end + self.takeoff_params["max_vel"] * t_decel + 0.5 * self.trajectory_acc_z * t_decel**2
            rospy.logdebug_throttle(1, f"减速阶段: t={t:.1f}s, z={self.trajectory_z:.2f}m, vel={self.trajectory_vel_z:.2f}m/s")

        # 阶段4：轨迹结束（已到达目标高度）
        else:
            self.trajectory_z = target_z
            self.trajectory_vel_z = 0.0
            self.trajectory_acc_z = 0.0

        # 3. 约束轨迹不超过目标高度（安全保护）
        self.trajectory_z = min(self.trajectory_z, target_z)

    def send_initial_setpoints(self):
        """提前发送100个初始位置指令，确保OFFBOARD模式可切换"""
        rospy.loginfo("发送初始位置指令（准备切换OFFBOARD模式）...")
        for i in range(100):
            setpoint = PoseStamped()
            setpoint.header.stamp = rospy.Time.now()
            setpoint.header.frame_id = "world"
            setpoint.pose.position = self.curr_pos  # 初始位置为当前位置
            self.pub_setpoint.publish(setpoint)
            rospy.sleep(0.05)  # 约20Hz发送
    
    def state_callback(self, msg):
        self.current_state = msg
        self.armed = msg.armed
        self.offboard = (msg.mode == "OFFBOARD")
    
    def odom_callback(self, msg):
        self.curr_pos = msg.pose.pose.position
        
        orientation_q = msg.pose.pose.orientation
        quaternion = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _, _, yaw = tf.transformations.euler_from_quaternion(quaternion)
        self.curr_yaw = yaw
    
    def arm_and_offboard(self):
        """处理模式切换和解锁，限制请求频率"""
        # 切换OFFBOARD模式（每5秒尝试一次）
        if not self.offboard and (rospy.Time.now() - self.last_req) > rospy.Duration(1.0):
            offb_set_mode = SetModeRequest()
            offb_set_mode.custom_mode = "OFFBOARD"
            if self.set_mode_client.call(offb_set_mode).mode_sent:
                rospy.loginfo("OFFBOARD模式已切换")
            self.last_req = rospy.Time.now()
        
        # 解锁（每5秒尝试一次）
        elif self.offboard and not self.armed and (rospy.Time.now() - self.last_req) > rospy.Duration(1.0):
            arm_cmd = CommandBoolRequest()
            arm_cmd.value = True
            if self.arm_client.call(arm_cmd).success:
                rospy.loginfo("无人机已解锁")
            self.last_req = rospy.Time.now()
    
    def check_takeoff_complete(self):
        """检查是否到达起飞位置"""
        dx = abs(self.curr_pos.x - 0)
        dy = abs(self.curr_pos.y - 0)
        dz = abs(self.curr_pos.z - self.takeoff_height)
        return (dx < self.takeoff_threshold) and (dy < self.takeoff_threshold) and (dz < self.takeoff_threshold)
    
    def publish_smooth_takeoff_cmd(self):
        """发布平滑轨迹指令（位置+速度+加速度）"""
        takeoff_cmd = PositionCommand()
        takeoff_cmd.header.stamp = rospy.Time.now()
        takeoff_cmd.header.frame_id = 'world'
        
        # XY方向保持原点（可根据需求添加XY微调）
        takeoff_cmd.position = Point(0, 0, self.trajectory_z)  # 实时更新的目标位置
        takeoff_cmd.velocity = Vector3(0, 0, self.trajectory_vel_z)  # 轨迹速度指令
        takeoff_cmd.acceleration = Vector3(0, 0, self.trajectory_acc_z)  # 轨迹加速度指令
        
        # 偏航角保持水平（无旋转）
        quaternion = [0, 0, 0, 1]
        _, _, yaw = tf.transformations.euler_from_quaternion(quaternion)
        takeoff_cmd.yaw = yaw
        takeoff_cmd.trajectory_flag = PositionCommand.TRAJECTORY_STATUS_READY  # 表示轨迹执行中
        self.pub_cmd.publish(takeoff_cmd)  # 假设pub_cmd是已初始化的发布者

    def timer_callback(self, event):
        # rospy.loginfo("timer...")
        if self.state == 0:
            # 等待初始化前的准备工作完成（如初始指令发送）
            self.state = 1
            rospy.loginfo("进入初始化阶段：切换OFFBOARD模式并解锁...")
        
        elif self.state == 1:
            # 执行模式切换和解锁
            if self.real_world_flag:
                rospy.loginfo("真实环境，请使用遥控器解锁")
                self.state = 2
            else:
                self.arm_and_offboard()
                if self.armed and self.offboard:
                    rospy.loginfo("初始化完成，进入起飞阶段")
                    self.state = 2
                    self.start_time = rospy.Time.now()
                else:
                    setpoint = PoseStamped()
                    setpoint.header.stamp = rospy.Time.now()
                    setpoint.header.frame_id = "world"
                    setpoint.pose.position = self.curr_pos  # 初始位置为当前位置
                    self.pub_setpoint.publish(setpoint)
        
        elif self.state == 2:
            if self.armed and self.offboard:
                # 实时生成平滑轨迹
                self.update_takeoff_trajectory()

                # 发布轨迹指令
                self.publish_smooth_takeoff_cmd()

                if self.check_takeoff_complete():
                    if not self.takeoff_complete:
                        self.start_time = rospy.Time.now()
                        self.takeoff_complete = True
                        if self.traj_mode == 1:
                            self.state = 3 # 直接进入圆形轨迹   
                            rospy.loginfo("已到达起飞位置，准备进入圆形轨迹")
                        elif self.traj_mode == 0:
                            rospy.loginfo("已到达起飞位置，开始定点")
  
            else:
                setpoint = PoseStamped()
                setpoint.header.stamp = rospy.Time.now()
                setpoint.header.frame_id = "world"
                setpoint.pose.position = self.curr_pos  # 初始位置为当前位置
                self.pub_setpoint.publish(setpoint)
                self.start_time = rospy.Time.now()
        elif self.state == 3:
            # 圆形轨迹逻辑
            t = (rospy.Time.now() - self.start_time).to_sec()
            
            # 平滑过渡到目标半径和速度
            if self.current_radius < self.radius:
                self.current_radius += self.radius / self.step
                self.current_velocity += self.velocity / self.step
                self.current_radius = min(self.current_radius, self.radius)
                self.current_velocity = min(self.current_velocity, self.velocity)
            
            # 计算角度
            if self.current_radius > 0.01:
                self.theta += (self.current_velocity / self.current_radius) * self.dt
            
            # 位置计算
            x = self.current_radius * np.cos(self.theta)
            y = self.current_radius * np.sin(self.theta)
            z = self.takeoff_height + self.z_amplitude * np.sin(2 * np.pi * self.z_frequency * t + self.z_phase)
            
            # 速度计算
            vx = -self.current_velocity * np.sin(self.theta) if self.current_radius > 0.01 else 0.0
            vy = self.current_velocity * np.cos(self.theta) if self.current_radius > 0.01 else 0.0
            vz = 2 * np.pi * self.z_frequency * self.z_amplitude * np.cos(2 * np.pi * self.z_frequency * t + self.z_phase)
            
            # 加速度计算
            ax = - (self.current_velocity**2 / self.current_radius) * np.cos(self.theta) if self.current_radius > 0.01 else 0.0
            ay = - (self.current_velocity**2 / self.current_radius) * np.sin(self.theta) if self.current_radius > 0.01 else 0.0
            az = - (2 * np.pi * self.z_frequency)**2 * self.z_amplitude * np.sin(2 * np.pi * self.z_frequency * t + self.z_phase)
            
            # 新增：计算过渡因子（0→1，与半径增长同步）
            transition_factor = min(self.current_radius / self.radius, 1.0) if self.radius > 0 else 0.0

            # yaw 偏移随过渡因子渐进叠加
            yaw = self.theta + (transition_factor * np.pi/2) if self.yaw_control else 0.0            
            yaw = normalize_yaw(yaw)
            # --------------------------
            # 新增：将yaw转换为四元数
            # 四元数表示：(qx, qy, qz, qw)
            # 仅考虑yaw（绕z轴旋转），roll和pitch设为0
            # --------------------------
            yaw_half = yaw / 2.0
            qw = np.cos(yaw_half)  # 实部
            qx = 0.0               # 绕x轴旋转分量（roll=0）
            qy = 0.0               # 绕y轴旋转分量（pitch=0）
            qz = np.sin(yaw_half)  # 绕z轴旋转分量（yaw）
            
            # 归一化四元数（数值计算可能引入微小误差）
            norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
            qx /= norm
            qy /= norm
            qz /= norm
            qw /= norm
            # 发布指令和轨迹
            self.publish_position_command(x, y, z, vx, vy, vz, ax, ay, az, yaw, qx, qy, qz, qw)
            self.publish_reference_trajectory(x, y, z)
    
    def publish_position_command(self, x, y, z, vx, vy, vz, ax, ay, az, yaw, qx, qy, qz, qw):
        msg = PositionCommand()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'world'
        msg.position = Point(x, y, z)
        msg.velocity = Vector3(vx, vy, vz)
        msg.acceleration = Vector3(ax, ay, az)
        msg.yaw = yaw
        msg.orientation.x = qx
        msg.orientation.y = qy
        msg.orientation.z = qz  
        msg.orientation.w = qw
        msg.trajectory_flag = PositionCommand.TRAJECTORY_STATUS_READY
        self.pub_cmd.publish(msg)
    
    def publish_reference_trajectory(self, x, y, z):
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = 'world'
        pose.pose.position = Point(x, y, z)
        self.reference_path.poses.append(pose)
        if len(self.reference_path.poses) > self.max_path_length:
            self.reference_path.poses.pop(0)
        self.reference_path.header.stamp = rospy.Time.now()
        self.pub_ref.publish(self.reference_path)

if __name__ == '__main__':
    try:
        CircleTrajectoryPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass