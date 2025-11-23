import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 样式设置
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号
plt.rcParams["lines.linewidth"] = 1.2
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["grid.linestyle"] = '--'


class LogVisualizer:
    def __init__(self):
        # 路径配置
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.log_file = os.path.join(self.script_dir, "controller_log.txt")
        
        # 数据存储
        self.data = pd.DataFrame(columns=[
            'timestamp', 'iterations', 'initial_error', 'final_error',
            'converged', 'opt_cost', 'all_cost', 'time_ms'
        ])
        self.time_intervals_ms = []
        self.metrics = {}
        
        # 2×2视图配置
        self.plot_config = {
            'opt_time': {
                'title': 'Optimization Time Analysis',
                'ylabel': 'Optimization Time (ms)',
                'data_col': 'opt_cost_ms',
                'avg_label': 'Average Optimization Time',
                'subplot': (2, 2, 1)  # 第一行第一列
            },
            'total_time': {
                'title': 'Total Time Analysis',
                'ylabel': 'Total Time (ms)',
                'data_col': 'all_cost_ms',
                'avg_label': 'Average Total Time',
                'subplot': (2, 2, 2)  # 第一行第二列
            },
            'iterations': {
                'title': 'Iteration Count Tracking',
                'ylabel': 'Number of Iterations',
                'data_col': 'iterations',
                'avg_label': 'Average Iterations',
                'subplot': (2, 2, 3)  # 第二行第一列
            },
            'errors': {
                'title': 'Error Comparison',
                'ylabel': 'Error Value',
                'initial_col': 'initial_error',
                'final_col': 'final_error',
                'subplot': (2, 2, 4)  # 第二行第二列
            }
        }
        
        # 颜色配置
        self.colors = {
            'data': 'royalblue',
            'average': 'darkorange',
            'initial_error': 'crimson',
            'final_error': 'forestgreen',
            'error_fill': 'lightblue'
        }
        
        # 初始化图形
        self.fig = plt.figure(figsize=(16, 14))
        self.fig.suptitle('Controller Performance Analysis', fontsize=16, y=0.98)

    def load_and_process_data(self):
        """加载数据并转换时间单位（忽略新增的input[0]字段）"""
        try:
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
            
            parsed_data = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = list(map(float, line.split()))
                
                # 关键修改：只取前7个字段，忽略新增的input[0]
                if len(parts) >= 7:  # 兼容新旧格式（旧格式7字段，新格式8字段）
                    parsed_data.append({
                        'timestamp': parts[0],
                        'iterations': int(parts[1]),
                        'initial_error': parts[2],
                        'final_error': parts[3],
                        'converged': int(parts[4]),
                        'opt_cost': parts[5],
                        'all_cost': parts[6]
                    })
                else:
                    print(f"Skipping invalid line: {line} (insufficient fields)")
            
            if not parsed_data:
                print("No valid data found in log file")
                return False
            
            # 数据处理与单位转换
            df = pd.DataFrame(parsed_data)
            df['timestamp_ms'] = df['timestamp'] * 1000  # 秒转毫秒
            df['time_ms'] = df['timestamp_ms'] - df['timestamp_ms'].min()  # 时间标准化
            df['opt_cost_ms'] = df['opt_cost'] * 1000  # 优化耗时转毫秒
            df['all_cost_ms'] = df['all_cost'] * 1000  # 总耗时转毫秒
            
            # 计算时间间隔
            if len(df) > 1:
                self.time_intervals_ms = df['timestamp_ms'].diff().dropna().tolist()
            
            self.data = df
            return True
        except Exception as e:
            print(f"Error processing data: {e}")
            return False

    def calculate_metrics(self):
        """计算性能指标"""
        if self.data.empty:
            return
        
        self.metrics = {
            'opt_time': {
                'mean': self.data['opt_cost_ms'].mean(),
                'max': self.data['opt_cost_ms'].max(),
                'min': self.data['opt_cost_ms'].min()
            },
            'total_time': {
                'mean': self.data['all_cost_ms'].mean(),
                'max': self.data['all_cost_ms'].max(),
                'min': self.data['all_cost_ms'].min()
            },
            'iterations': {
                'mean': self.data['iterations'].mean(),
                'max': self.data['iterations'].max(),
                'min': self.data['iterations'].min()
            },
            'errors': {
                'initial': {'mean': self.data['initial_error'].mean()},
                'final': {'mean': self.data['final_error'].mean()},
                'reduction': self.data['initial_error'].mean() - self.data['final_error'].mean()
            }
        }
        
        if self.time_intervals_ms:
            self.metrics['time_interval'] = {
                'mean': np.mean(self.time_intervals_ms),
                'max': np.max(self.time_intervals_ms)
            }
        
        self.metrics['convergence_rate'] = self.data['converged'].mean() * 100

    def print_metrics(self):
        """打印性能指标"""
        if not self.metrics:
            print("No metrics available")
            return
        
        print("\n=== Controller Performance Metrics ===")
        print(f"\nOptimization Time (ms):")
        print(f"  Mean: {self.metrics['opt_time']['mean']:.2f}, Max: {self.metrics['opt_time']['max']:.2f}, Min: {self.metrics['opt_time']['min']:.2f}")
        
        print(f"\nTotal Time (ms):")
        print(f"  Mean: {self.metrics['total_time']['mean']:.2f}, Max: {self.metrics['total_time']['max']:.2f}, Min: {self.metrics['total_time']['min']:.2f}")
        
        print(f"\nIterations:")
        print(f"  Mean: {self.metrics['iterations']['mean']:.2f}, Max: {self.metrics['iterations']['max']}, Min: {self.metrics['iterations']['min']}")
        
        print(f"\nError Analysis:")
        print(f"  Initial Error (Mean): {self.metrics['errors']['initial']['mean']:.4f}")
        print(f"  Final Error (Mean): {self.metrics['errors']['final']['mean']:.4f}")
        print(f"  Average Error Reduction: {self.metrics['errors']['reduction']:.4f}")
        
        if 'time_interval' in self.metrics:
            print(f"\nSampling Interval (ms):")
            print(f"  Mean: {self.metrics['time_interval']['mean']:.2f} ms, Max: {self.metrics['time_interval']['max']:.2f} ms")
        print(f"\nConvergence Rate: {self.metrics['convergence_rate']:.2f}%")

    def plot_with_average(self, ax, x_data, y_data, mean_value, title, ylabel, avg_label):
        """绘制带平均线的曲线"""
        ax.plot(x_data, y_data, color=self.colors['data'], label='Measured Value')
        ax.axhline(y=mean_value, color=self.colors['average'], linestyle='--', label=avg_label)
        
        ax.text(0.02, 0.95, f"Avg: {mean_value:.2f} ms", 
                transform=ax.transAxes, 
                bbox=dict(facecolor='white', alpha=0.8),
                fontsize=9)
        
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Time (ms)')
        ax.grid(True)
        ax.legend(loc='upper right')

    def plot_error_comparison(self, ax, x_data, initial_data, final_data, title, ylabel):
        """绘制误差对比曲线"""
        ax.plot(x_data, initial_data, color=self.colors['initial_error'], label='Initial Error')
        ax.plot(x_data, final_data, color=self.colors['final_error'], label='Final Error')
        ax.fill_between(x_data, initial_data, final_data, color=self.colors['error_fill'], alpha=0.3, label='Error Reduction')
        
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Time (ms)')
        ax.grid(True)
        ax.legend(loc='upper right')

    def create_visualization(self):
        """创建2×2布局的可视化图表"""
        if self.data.empty:
            return
        
        # 1. 优化耗时曲线（第一行第一列）
        cfg = self.plot_config['opt_time']
        ax1 = self.fig.add_subplot(*cfg['subplot'])
        self.plot_with_average(
            ax=ax1,
            x_data=self.data['time_ms'],
            y_data=self.data[cfg['data_col']],
            mean_value=self.metrics['opt_time']['mean'],
            title=cfg['title'],
            ylabel=cfg['ylabel'],
            avg_label=cfg['avg_label']
        )

        # 2. 总耗时曲线（第一行第二列）
        cfg = self.plot_config['total_time']
        ax2 = self.fig.add_subplot(*cfg['subplot'])
        self.plot_with_average(
            ax=ax2,
            x_data=self.data['time_ms'],
            y_data=self.data[cfg['data_col']],
            mean_value=self.metrics['total_time']['mean'],
            title=cfg['title'],
            ylabel=cfg['ylabel'],
            avg_label=cfg['avg_label']
        )

        # 3. 迭代次数曲线（第二行第一列）
        cfg = self.plot_config['iterations']
        ax3 = self.fig.add_subplot(*cfg['subplot'])
        self.plot_with_average(
            ax=ax3,
            x_data=self.data['time_ms'],
            y_data=self.data[cfg['data_col']],
            mean_value=self.metrics['iterations']['mean'],
            title=cfg['title'],
            ylabel=cfg['ylabel'],
            avg_label=cfg['avg_label']
        )

        # 4. 误差对比曲线（第二行第二列）
        cfg = self.plot_config['errors']
        ax4 = self.fig.add_subplot(*cfg['subplot'])
        self.plot_error_comparison(
            ax=ax4,
            x_data=self.data['time_ms'],
            initial_data=self.data[cfg['initial_col']],
            final_data=self.data[cfg['final_col']],
            title=cfg['title'],
            ylabel=cfg['ylabel']
        )

        # 添加全局信息
        if 'time_interval' in self.metrics:
            footer_text = f"Average Sampling Interval: {self.metrics['time_interval']['mean']:.2f} ms | Convergence Rate: {self.metrics['convergence_rate']:.2f}%"
            self.fig.text(0.5, 0.01, footer_text, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

        # 调整布局
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    def show_and_save(self):
        """显示并保存图表"""
        plt.show()
        self.fig.savefig('controller_2x2_analysis.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved as 'controller_2x2_analysis.png'")

    def run(self):
        """执行数据分析流程"""
        if self.load_and_process_data():
            self.calculate_metrics()
            self.print_metrics()
            self.create_visualization()
            self.show_and_save()


if __name__ == "__main__":
    visualizer = LogVisualizer()
    visualizer.run()