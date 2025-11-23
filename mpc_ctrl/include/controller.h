/*************************************************************/
/* Acknowledgement: github.com/uzh-rpg/rpg_quadrotor_control */
/*************************************************************/

#ifndef __CONTROLLER_H
#define __CONTROLLER_H

#include <fstream>
#include <mavros_msgs/AttitudeTarget.h>
#include <quadrotor_msgs/Px4ctrlDebug.h>
#include <queue>

#include "JPCM.h"
#include "input.h"
#include <Eigen/Dense>
#include "factors.h"
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Quaternion.h>
#include <gtsam/geometry/Rot3.h>
#include "type.h"

#include <Eigen/StdVector>
class Controller
{
public:
  using Dist_Dou = std::normal_distribution<double>;

  Controller(Parameter_t &);

  ~Controller();
  // 使用 GTSAM 库（因子图优化库）融合多传感器数据，估计无人机状态（四旋翼状态估计核心模块）
  quadrotor_msgs::Px4ctrlDebug fusion(const Odom_Data_t &odom, const Imu_Data_t &imu_raw, const Odom_Data_t &GT, gtsam::Pose3 &fus_pose, gtsam::Vector3 &fus_vel, gtsam::Vector3 &fus_w);
  // 从静止或低速状态的传感器数据中估计初始状态，为控制提供初始值（控制器启动必需步骤）
  bool initializeState(const std::vector<Imu_Data_t> &imu_raw, const std::vector<Odom_Data_t> &fakeGPS, gtsam::Vector3 &init_vel, gtsam::Vector6 &init_bias);

  // 输入：期望状态（des，如位置、速度、姿态）、估计的里程计状态（odom）、IMU 数据（imu）。
  // 输出：调试消息和控制指令（thr_bodyrate_u，含推力和机体角速度指令）。
  // 功能：实现基础反馈控制，计算无人机的推力和角速度指令
  // DFBC（基于模型的控制）模式下使用。
  quadrotor_msgs::Px4ctrlDebug calculateControl(const Desired_State_t &des, const Odom_Data_t &odom, const Imu_Data_t &imu, Controller_Output_t &thr_bodyrate_u);

  // 输入：增加地面真值（GT）和控制模式切换参数（mode_switch）。
  // 功能：基于模型预测控制优化轨迹跟踪，支持控制模式切换（如从位置控制切换到姿态控制）。
  // SP-MPC
  quadrotor_msgs::Px4ctrlDebug calculateControl(const Desired_State_t &des, const Odom_Data_t &GT, const Odom_Data_t &odom, const Imu_Data_t &imu,
                                                Controller_Output_t &thr_bodyrate_u, CTRL_MODE mode_switch);

  // 输入：增加原始 IMU 数据（imu_raw），可能用于噪声补偿或鲁棒控制。
  // 功能：联合优化位置和姿态控制，适用于高动态场景。
  // JPCM（联合预测控制模型）模式下使用。
  quadrotor_msgs::Px4ctrlDebug calculateControl(const Desired_State_t &des, const Odom_Data_t &odom, const Imu_Data_t &imu, const Imu_Data_t &imu_raw,
                                                Controller_Output_t &thr_bodyrate_u, CTRL_MODE mode_switch);

  // 功能：结合期望加速度和当前速度，计算无人机所需的总推力（考虑速度对推力的影响，如空气阻力补偿）。
  double computeDesiredCollectiveThrustSignal(const Eigen::Vector3d &des_acc, const Eigen::Vector3d &v);

  // 功能：简化版推力计算，直接根据期望加速度计算推力（忽略速度影响，基础控制场景使用）
  double computeDesiredCollectiveThrustSignal(const Eigen::Vector3d &des_acc);

  // 功能：在线估计推力与加速度的映射关系（如校准推力系数，补偿电机非线性）
  bool estimateThrustModel(const Eigen::Vector3d &est_v, const Parameter_t &param);

  // 功能：重置推力到加速度的映射参数（如传感器校准后或控制模式切换时使用）
  void resetThrustMapping(void);

  // 输入：悬停推力（hover_thrust）。
  // 功能：计算推力到加速度的转换系数（thr2acc_），公式为重力加速度 / 悬停推力（核心参数，用于推力与加速度的线性映射）。
  void set_hover_thrust(float hover_thrust) { thr2acc_ = param_.gra / hover_thrust; }
  double rpy_from_quaternion(const Eigen::Quaterniond &q);
  double normalize_yaw(double yaw);
  bool almostZero(const double value) const;
  bool almostZeroThrust(const double thrust_value) const;
  Eigen::Vector3d computeNominalReferenceInputs(
      const Eigen::Quaterniond &attitude_estimate, const Desired_State_t &des) const;
  Eigen::Quaterniond computeDesiredAttitude(
      const Eigen::Vector3d &desired_acceleration, const double reference_heading,
      const Eigen::Quaterniond &attitude_estimate) const;
  Eigen::Vector3d computeRobustBodyXAxis(
      const Eigen::Vector3d &x_B_prototype, const Eigen::Vector3d &x_C,
      const Eigen::Vector3d &y_C,
      const Eigen::Quaterniond &attitude_estimate) const;
  Odom_Data_t add_Guassian_noise(const Odom_Data_t &odom);

  const Parameter_t &get_param() { return param_; };

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  const uint16_t IDX_P_START = 100;

  Parameter_t param_;
  quadrotor_msgs::Px4ctrlDebug debug_msg_;
  std::queue<std::pair<ros::Time, double>> timed_thrust_;
  static constexpr double kMinNormalizedCollectiveThrust_ = 3.0; // 功能：限制推力下限，避免推力过小导致无人机失控（物理约束，电机最小推力限制）

  // Thrust-accel mapping params
  const double rho2_ = 0.998; // 滤波器系数（固定值，可能用于推力数据的低通滤波）
  double thr2acc_;            // 推力到加速度的转换系数（由set_hover_thrust初始化）
  double P_;                  // 可能为推力模型估计的协方差或增益参数
  std::vector<Odom_Data_t> odom_data_v_;
  std::vector<Odom_Data_t> odom_data_noise_;
  std::vector<Imu_Data_t> imu_data_v_;
  std::vector<Desired_State_t> des_data_v_;

  double fromQuaternion2yaw(Eigen::Quaterniond q);
  double limit_value(double upper_bound, double input, double lower_bound);
  Eigen::Vector3d limit_err(const Eigen::Vector3d err, const double p_err_max);
  // 指向buildJPCMFG类实例，用于构建状态估计的因子图（GTSAM 核心组件）
  std::shared_ptr<buildJPCMFG> FGbuilder;
  uint64_t state_idx_ = 0;       // 状态变量索引计数器
  gtsam_fg graph_;               // GTSAM 因子图对象（存储约束因子）
  gtsam_sols initial_value_;     // 优化的初始猜测值
  bool init_state_flag_ = false; // 状态初始化完成标志

  gtsam::Vector3 init_vel_;
  gtsam::Vector6 init_bias_;

  double dt_;
  uint16_t opt_traj_lens_;   // 优化轨迹长度（MPC 预测时域）
  uint16_t window_lens_ = 5; // 滑动窗口长度（用于滚动优化）

  Dist_Dou position_noise_x;
  Dist_Dou rotation_noise_x;
  Dist_Dou velocity_noise_x;

  Dist_Dou position_noise_y;
  Dist_Dou rotation_noise_y;
  Dist_Dou velocity_noise_y;

  Dist_Dou position_noise_z;
  Dist_Dou rotation_noise_z;
  Dist_Dou velocity_noise_z;

protected:
  std::ofstream log_;
};

#endif
