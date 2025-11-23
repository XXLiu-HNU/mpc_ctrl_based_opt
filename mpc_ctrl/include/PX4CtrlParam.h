#ifndef __PX4CTRLPARAM_H
#define __PX4CTRLPARAM_H

#include <ros/ros.h>

class Parameter_t
{
public:
	struct Gain
	{
		double Kp0, Kp1, Kp2;
		double Kv0, Kv1, Kv2;
		double Kvi0, Kvi1, Kvi2;
		double Kvd0, Kvd1, Kvd2;
		double KAngR, KAngP, KAngY;
		double PErrMax, VErrMax;
	};

	struct RotorDrag
	{
		double x, y, z;
		double k_thrust_horz;
	};

	struct MsgTimeout
	{
		double odom;
		double rc;
		double cmd;
		double imu;
		double bat;
	};
	// 无人机推力与油门的关系通常是非线性的（受电机、螺旋桨特性影响），常用二次多项式模型描述：
	// F = K1 * u + K2 * u^2 + K3
	// 其中 F 是推力，u 是油门百分比（0-100%），K1、K2、K3 是模型参数。
	// 这里的 K1、K2、K3 是通过实验或仿真得到的系数，用于将油门信号转换为推力。
	struct ThrustMapping
	{
		bool print_val;
		double K1;
		double K2;
		double K3;
		bool accurate_thrust_model;
		double hover_percentage;   // 无人机悬停时的油门指令百分比（归一化值，通常范围 0~1 或 0%~100%）
		double thrust_upper_bound; // 出于安全和硬件保护，限制无人机输出的最大推力
		double thrust_lower_bound; // 该值通常大于 0（例如设置为无人机重力的 50%），但在着陆阶段可能允许更低值（如接近地面时逐步减小到 0）
	};

	struct RCReverse
	{
		bool roll;
		bool pitch;
		bool yaw;
		bool throttle;
	};

	struct AutoTakeoffLand
	{
		bool enable;
		bool enable_auto_arm;
		bool no_RC;
		double height;
		double speed;
	};

	// factor graph params
	struct FactorGraph
	{
		std::string LOG_NAME;

		double PRI_VICON_POS_COV;
		double PRI_VICON_VEL_COV;

		double CONTROL_P_COV_X;
		double CONTROL_P_COV_Y;
		double CONTROL_P_COV_Z;
		double CONTROL_PF_COV_X;
		double CONTROL_PF_COV_Y;
		double CONTROL_PF_COV_Z;
		double CONTROL_V_COV;
		double DYNAMIC_P_COV;
		double DYNAMIC_R_COV;
		double DYNAMIC_V_COV;
		double CONTROL_R1_COV;
		double CONTROL_R2_COV;
		double CONTROL_R3_COV;

		double INPUT_JERK_T;
		double INPUT_JERK_M;
		double INPUT_JERK_M3;

		int OPT_LENS_TRAJ;
		int WINDOW_SIZE;

		double high;
		double low;
		double thr;

		double ghigh;
		double glow;
		double gthr;

		double alpha;

		double POS_MEAS_MEAN;
		double POS_MEAS_COV;
		double VEL_MEAS_COV;
		double ROT_MEAS_COV;

		double PRIOR_POS_MEAS_COV;
		double PRIOR_VEL_MEAS_COV;
		double PRIOR_ROT_MEAS_COV;

		double acc_sigma_x;
		double acc_bias_imu_x;
		double acc_sigma_y;
		double acc_bias_imu_y;
		double acc_sigma_z;
		double acc_bias_imu_z;

		double gyro_sigma_x;
		double gyro_bias_sigma_x;
		double gyro_sigma_y;
		double gyro_bias_sigma_y;
		double gyro_sigma_z;
		double gyro_bias_sigma_z;

		double prior_acc_sigma;
		double prior_gyro_sigma;

		bool opt_gravity_rot;
		bool use_vel;
		bool use_rot;
	};

	Gain gain;
	RotorDrag rt_drag;
	MsgTimeout msg_timeout;
	RCReverse rc_reverse;
	ThrustMapping thr_map;
	AutoTakeoffLand takeoff_land;
	FactorGraph factor_graph;
	int ctrl_mode;

	int pose_solver;
	double mass;
	double gra;
	double max_angle;
	double ctrl_freq_max;
	double max_manual_vel;
	double low_voltage;
	int odom_freq;

	bool use_bodyrate_ctrl;
	std::string output_log_path;

	Parameter_t();
	void config_from_ros_handle(const ros::NodeHandle &nh);
	void config_full_thrust(double hov);

private:
	template <typename TName, typename TVal>
	void read_essential_param(const ros::NodeHandle &nh, const TName &name, TVal &val)
	{
		if (nh.getParam(name, val))
		{
			// pass
		}
		else
		{
			ROS_ERROR_STREAM("Read param: " << name << " failed.");
			ROS_BREAK();
		}
	};
};

#endif