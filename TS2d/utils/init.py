import pandas as pd
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import re
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import matplotlib
from matplotlib.gridspec import GridSpec

matplotlib.rcParams['animation.writer'] = 'pillow'
# 读取CSV文件，跳过前8行文字
# header=None 表示不将第一行作为列名


def Plot_two(x,y,T,P,output_dir = None,a = None):
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)

        # 确保xy方向的范围一致
        xy_range = max(x_max - x_min, y_max - y_min)
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        x_min = x_center - xy_range / 2
        x_max = x_center + xy_range / 2
        y_min = y_center - xy_range / 2
        y_max = y_center + xy_range / 2

        # 创建网格
        grid_resolution = 2000  # 网格分辨率
        xi = np.linspace(x_min, x_max, grid_resolution)
        yi = np.linspace(y_min, y_max, grid_resolution)
        xi, yi = np.meshgrid(xi, yi)

        # 计算到原点的距离
        r = np.sqrt(xi ** 2 + yi ** 2)
        r_values = np.sqrt(x ** 2 + y ** 2)
        min_r = np.min(r_values)
        max_r = np.max(r_values)

        # 创建统一的环形掩码
        ring_mask = (r >= min_r) & (r <= max_r)

        # 1. 温度分布子图
        # 使用线性插值创建连续温度场
        Ti = griddata((x, y), T, (xi, yi), method='cubic', fill_value=np.nan)
        Ti_masked = np.where(ring_mask, Ti, np.nan)

        # 确定统一的值范围
        T_min, T_max = np.nanmin(Ti_masked), np.nanmax(Ti_masked)
        P_min, P_max = np.nanmin(P), np.nanmax(P)

        # 绘制温度分布
        im1 = axes[0].imshow(Ti_masked,
                             extent=[x_min, x_max, y_min, y_max],
                             origin='lower',
                             cmap='jet',
                             aspect='auto',  # 使用auto保持比例
                             alpha=0.65,
                             vmin=T_min,
                             vmax=T_max)
        #
        # axes[0].set_title('Temperature Distribution', fontsize=20,
        #                  pad=20)

        axes[0].axis('off')
        # axes[0].set_xlim(-1.65, 1.65)
        # axes[0].set_ylim(-1.65, 1.65)
        axes[0].set_aspect('equal')
        cbar1 = plt.colorbar(im1, ax=axes[0])
        cbar1.ax.tick_params(labelsize=24)

        # 2. 压强分布子图
        # 使用线性插值创建连续压强场
        Pi = griddata((x, y), P, (xi, yi), method='cubic', fill_value=np.nan)
        Pi_masked = np.where(ring_mask, Pi, np.nan)

        # 绘制压强分布
        im2 = axes[1].imshow(Pi_masked,
                             extent=[x_min, x_max, y_min, y_max],
                             origin='lower',
                             cmap='Spectral',
                             aspect='auto',  # 使用auto保持比例
                             alpha=0.8,
                             vmin=np.nanmin(Pi_masked),
                             vmax=np.nanmax(Pi_masked))

        # axes[1].set_title(r'$\sigma^v$ Distribution', fontsize=20,
        #                  pad=20)
        axes[1].axis('off')
        # axes[1].set_xlim(-1.65, 1.65)
        # axes[1].set_ylim(-1.65, 1.65)
        axes[1].set_aspect('equal')

        # 添加颜色条
        cbar2 = plt.colorbar(im2, ax=axes[1])
        cbar2.ax.tick_params(labelsize=24)
        # 确保两个子图有相同的坐标轴范围
        axes[0].set_aspect('equal')
        axes[1].set_aspect('equal')

        # 调整布局
        plt.tight_layout()

        # 保存图片
        # plt.show()
        output_path = os.path.join(output_dir, f'{a}.svg')
        plt.savefig(output_path, bbox_inches='tight')
        plt.close(fig)

def load_normalized_data(npz_file_path):
    """加载归一化后的数据"""
    print(f"Loading normalized data from: {npz_file_path}")
    data = np.load(npz_file_path, allow_pickle=True)
    samples = data['samples']
    print(f"Loaded {len(samples)} samples")
    return samples


def extract_same_location_data(samples, target_cx=0.85, target_cy=0.6, tolerance=1e-6):

    filtered_samples = []

    for sample in samples:
        params = sample['params']
        cx, cy, t = params

        # 检查是否匹配目标位置
        if (abs(cx - target_cx) < tolerance and
                abs(cy - target_cy) < tolerance):
            filtered_samples.append(sample)

    # 按时间步排序
    filtered_samples.sort(key=lambda x: x['params'][2])

    # 提取所有唯一时间步
    unique_times = []
    for sample in filtered_samples:
        t = sample['params'][2]
        if abs(t) > 0.003:  # 排除接近0.00的时刻
            if not unique_times or abs(t - unique_times[-1]) > tolerance:
                unique_times.append(t)

    print(f"Found {len(filtered_samples)} samples at location ({target_cx}, {target_cy})")
    print(f"Time range: {unique_times[0]:.4f} to {unique_times[-1]:.4f}")

    return filtered_samples, unique_times
def visualize_specific_location(npz_file_path, target_cx=0.85, target_cy=0.6, output_base_dir=None):

    samples = load_normalized_data(npz_file_path)

    # 2. 提取特定位置的数据
    print(f"\nExtracting data for location: c_x={target_cx}, c_y={target_cy}")
    filtered_samples, unique_times = extract_same_location_data(
        samples, target_cx, target_cy
    )

    if not filtered_samples:
        print(f"No data found for location ({target_cx}, {target_cy})")
        return

    # 3. 创建输出目录
    if output_base_dir is None:
        # 从输入文件路径推断输出目录
        base_dir = os.path.dirname(npz_file_path)
        output_dir = os.path.join(
            base_dir,
            f'visualization_cx_{target_cx:.2f}_cy_{target_cy:.2f}'
        )
    else:
        output_dir = os.path.join(
            output_base_dir,
            f'visualization_cx_{target_cx:.2f}_cy_{target_cy:.2f}'
        )

    os.makedirs(output_dir, exist_ok=True)

    # 4. 保存时间步信息
    time_info_path = os.path.join(output_dir, 'time_steps.txt')
    with open(time_info_path, 'w') as f:
        f.write(f"Location: c_x={target_cx}, c_y={target_cy}\n")
        f.write(f"Total time steps: {len(filtered_samples)}\n")
        f.write("\nTime steps:\n")
        for i, t in enumerate(unique_times):
            f.write(f"Frame {i:04d}: t = {t:.6f}\n")

    print(f"\nTime steps information saved to: {time_info_path}")

    # 5. 创建每个时间步的可视化
    print("\nCreating individual time step visualizations...")
    x,y,T,P = create_visualization(filtered_samples, output_dir)
    return x, y, T, P
def create_visualization(samples, output_dir, prefix='frame'):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    from tqdm import tqdm

    os.makedirs(output_dir, exist_ok=True)

    # 设置绘图样式
    plt.style.use('default')

    # 为每个样本（时间步）创建图片
    for i, sample in enumerate(tqdm(samples, desc="Creating visualizations")):
        params = sample['params']
        cx, cy, t = params
        if abs(t-0.77)<1e-4:

            x = sample['x']  # 变形后的x坐标
            y = sample['y']  # 变形后的y坐标
            T = sample['T']  # 归一化后的温度
            P = sample['P']  # 归一化后的压强
        else:
            continue

    return x, y, T, P
df = pd.read_csv("D:\desktop\output-special.csv", header=None, skiprows=9)

# 方法1：获取前两列（索引0和1）的数据
array1 = df.iloc[:, [0, 1]].to_numpy()  # 第1-2列
# 方法2：获取307-311列的数据（注意Python是0-based索引，所以307列对应索引306）
# 如果CSV的列是从1开始编号的：
array2 = df.iloc[:, 310:314].to_numpy()  # 索引306-310对应第307-311列
array2[:, 1] = (array2[:, 1] - 18093686784) / 15298610176
array2[:, 0] = (array2[:, 0] - 312.5865478515625) / 49.418724060058594

array2[:, 2] = array2[:, 2] + array1[:, 0]
array2[:, 3] += array1[:, 1]
T = array2[:, 0]
P = array2[:, 1]
x = array2[:, 2]
y = array2[:, 3]


normalized_npz_path = r"D:\desktop\output861\TS_data_normalized.npz"  # 你需要先运行归一化函数
target_cx = 0.5
target_cy = -0.65
x1, y1, T1, P1 = visualize_specific_location(
        npz_file_path=normalized_npz_path,
        target_cx=target_cx,
        target_cy=target_cy
    )
Plot_two(x, y, T, P,output_dir='D:\\desktop\\output861\\vis_folder',a = 'pred_inv')
Plot_two(x, y, T1, P1,output_dir='D:\\desktop\\output861\\vis_folder',a = 'True_inv')
Plot_two(x, y, np.abs(T-T1), np.abs(P-P1),output_dir='D:\\desktop\\output861\\vis_folder',a = 'error_inv')