import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import os


# def plot_performance_improvement(per, categories):
#     """
#     绘制性能提升百分比的雷达图
#
#     参数:
#     per -- 性能提升百分比数组 (范围通常在-100%到+100%之间)
#     """
#     assert len(per) == 8, "需要8个性能提升值"
#
#     # 角度设置
#     num_vars = len(per)
#     angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
#
#     # 闭合图形
#     per = np.append(per, per[0])
#     angles += angles[:1]
#
#     # 创建雷达图
#     plt.figure(figsize=(10, 8))
#     plt.gca().set_facecolor('none')  # 设置坐标区背景透明
#     plt.gcf().set_facecolor('none')  # 设置图形背景透明
#     ax = plt.subplot(111, polar=True)
#
#     # 绘制填充区域 (透明度0.2)
#     ax.fill(angles, per, 'g', alpha=0.2, label='Performance Improvement')
#
#     # 绘制边界线
#     ax.plot(angles, per, 'g', linewidth=2, linestyle='solid')
#
#     # 设置坐标轴范围
#     min_val = min(per) - 1  # 留出5%的边距
#     max_val = max(per) + 1  # 留出5%的边距
#     ax.set_ylim(min_val, max_val)
#
#     # 获取当前y轴刻度并转换为整数
#     yticks = ax.get_yticks()
#     yticks = [int(tick) for tick in yticks]  # 转换为整数
#
#     # 只保留每隔一个的刻度标签，但确保包含最大值
#     visible_labels = []
#     for i, tick in enumerate(yticks):
#         if i % 2 == 1 :  # 每隔一个或最大值
#             visible_labels.append(f"{tick}%")  # 添加百分号
#         else:
#             visible_labels.append('')  # 空字符串表示不显示标签
#
#     # 设置刻度标签
#     ax.set_yticks(yticks)  # 确保刻度位置正确
#     ax.set_yticklabels(visible_labels, fontsize=30)  # 应用自定义标签
#
#     # 添加类别标签
#     ax.set_xticklabels([])
#
#     # 添加网格和美化
#     ax.grid(True, linestyle='-.',  linewidth=1.5)
#     ax.spines['polar'].set_visible(False)
#     ax.set_theta_offset(np.pi / 2)
#     ax.set_theta_direction(-1)
#
#     plt.tight_layout()
#     output_path = 'D:\\desktop\\stefan_plots\\result'
#     plot_filename = os.path.join(output_path, 'rada.svg')
#     plt.savefig(plot_filename, dpi=300)
#     plt.show()
#
#
# # 示例使用
# if __name__ == "__main__":
#     # 示例数据（8个误差值）
#     errors_A = np.array([0.000818, 0.008803, 0.004253, 0.000331, 0.001482, 0.027596, 0.012980, 0.001021])
#     errors_B = np.array([0.005611, 0.003403, 0.003755, 0.000256, 0.041680, 0.537011, 0.510699, 0.033763])
#     per = (errors_B - errors_A) / errors_B * 100  # 转换为百分比
#     print(per)
#     labels = [
#         'Accuracy', 'Precision', 'Recall', 'F1-Score',
#         'Robustness', 'Speed', 'Scalability', 'Stability'
#     ]
#
#     plot_performance_improvement(per, labels)

import matplotlib.pyplot as plt
import os
from tensorboard.backend.event_processing import event_accumulator


def plot_grouped_tensorboard_scalars(log_dirs, scalar_groups, group_names, labels=None,
                                     colors=None, xlabel='Steps', ylabels=None,
                                     figsize=(10, 6), legend=True, save_dir=None,
                                     dpi=150, skip_initial=0.01):
    """
    参数:
    log_dirs: TensorBoard日志目录列表
    scalar_groups: 分组后的标量名称列表的列表
    group_names: 每组标量的名称列表
    labels: 每条曲线的标签列表
    colors: 每条曲线的Hex颜色代码列表，格式为['#RRGGBB', ...]
    xlabel: x轴标签
    ylabels: 每组标量的y轴标签列表
    figsize: 图表大小
    legend: 是否显示图例
    save_dir: 保存目录
    dpi: 图片分辨率
    skip_initial: 跳过初始数据的比例
    """

    if labels is None:
        labels = [f'Model {i + 1}' for i in range(len(log_dirs))]
    elif len(labels) != len(log_dirs):
        raise ValueError("labels的长度必须与log_dirs相同")

    if ylabels is None:
        ylabels = ['Value'] * len(scalar_groups)
    elif len(ylabels) != len(scalar_groups):
        raise ValueError("ylabels的长度必须与scalar_groups相同")

    figures = []
    color_idx = 0  # 颜色索引

    for group_idx, (scalar_names, group_name) in enumerate(zip(scalar_groups, group_names)):
        plt.figure(figsize=figsize)
        ax = plt.gca()  # 获取当前坐标轴

        for scalar_name in scalar_names:
            for log_dir, label in zip(log_dirs, labels):
                try:
                    # 加载事件数据
                    ea = event_accumulator.EventAccumulator(log_dir)
                    ea.Reload()

                    # 检查标量是否存在
                    full_scalar_name = f"Test_l2/{scalar_name}"  # 添加前缀
                    if full_scalar_name not in ea.Tags()['scalars']:
                        available = ', '.join(ea.Tags()['scalars'])
                        print(f"警告: '{full_scalar_name}'不在日志 {log_dir} 的可用标量中。可用标量: {available}")
                        continue

                    scalar_data = ea.Scalars(full_scalar_name)

                    # 提取步骤和值
                    steps = [s.step for s in scalar_data]
                    values = [s.value for s in scalar_data]

                    # 计算要跳过的数据点数量
                    skip_points = int(len(steps) * skip_initial)

                    # 跳过初始数据点
                    steps = steps[skip_points:]
                    values = values[skip_points:]

                    # 获取当前颜色
                    if colors and color_idx < len(colors):
                        color = colors[color_idx]
                    else:
                        color = None  # 让matplotlib自动选择颜色

                    # 绘制曲线
                    plt.plot(steps, values,  color=color)
                    color_idx += 1  # 移动到下一个颜色

                except Exception as e:
                    print(f"处理日志目录 '{log_dir}' 时出错: {str(e)}")
                    continue

        # 重置颜色索引为下一组
        color_idx = 0

        # 设置图表属性

        plt.yscale('log')

        # 去除网格和边框
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_linewidth(0.1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)

        # 设置坐标轴刻度标签大小
        ax.tick_params(axis='both', which='major', labelsize=12 )



        # 保存图片
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{group_name.replace(' ', '_')}.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=300, transparent=True)
            print(f"图表已保存至: {save_path}")

        figures.append(plt.gcf())

    return figures


# 使用示例
log_dirs = [
    "D:\\desktop\\BCO\\tensorboard\\5_271\\PI---0528_013857\\events.out.tfevents.1748367537.WIN-GHLON8JDP80.33452.0",
    "D:\\desktop\\BCO\\tensorboard\\5_271\\supervise---0528_180545\\events.out.tfevents.1748426745.WIN-GHLON8JDP80.35180.0",
]

# 分组后的标量名称
scalar_groups = [
    ['phi', 'rho'],  # 第一组
    ['v', 'T']  # 第二组
]

group_names = ['Phi and Rho Comparison', 'V and T Comparison']
labels = ['Model A', 'Model B']
ylabels = ['Loss (log scale)', 'Value (log scale)']

custom_colors = [
    '#B2B2FF',  # Model A phi
    '#FFB2B2',  # Model A rho
    '#66FF66',  # Model B phi
    '#FF66FF',  # Model B rho

    '#B2D9B2',  # Model A v
    '#FFE4B2',  # Model A T
    '#C21E1E',  # Model B v
    '#FF66B2',  # Model B T
]

# 调用函数
plot_grouped_tensorboard_scalars(
    log_dirs=log_dirs,
    scalar_groups=scalar_groups,
    group_names=group_names,
    labels=labels,
    colors=custom_colors,
    ylabels=ylabels,
    figsize=(3, 3),
    save_dir='D:\\desktop\\stefan_plots\\result'
)

