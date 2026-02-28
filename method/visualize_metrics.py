# -*- coding: utf-8 -*-
"""
指标可视化：AoI、Coverage、能耗、数据量等对比图
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# 中文字体与负号
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_episode_metrics(all_episode_stats, output_dir, prefix=''):
    """
    绘制多轮测试中各指标随 episode 的变化（折线+柱状）。

    all_episode_stats: list of (reward, avg_return, mean_aoi, mean_energy, collected_data, coverage)
                       每项对应一轮 run_k_episodes(k=1) 的 statistics
    """
    _ensure_dir(output_dir)
    n = len(all_episode_stats)
    if n == 0:
        return

    rewards, returns, aois, energies, data_amounts, coverages = zip(*all_episode_stats)
    episodes = np.arange(1, n + 1)

    # 2x2 子图：AoI, Coverage, Energy, Data amount
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('Metrics over Episodes' + (' — ' + prefix if prefix else ''), fontsize=12)

    axes[0, 0].bar(episodes - 0.2, aois, width=0.4, label='Mean AoI', color='steelblue', alpha=0.8)
    axes[0, 0].set_ylabel('Mean AoI')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_title('Mean AoI per Episode')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].bar(episodes - 0.2, coverages, width=0.4, label='User Coverage', color='coral', alpha=0.8)
    axes[0, 1].set_ylabel('Coverage')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_title('User Coverage per Episode')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].bar(episodes - 0.2, energies, width=0.4, label='Energy (1 - residual)', color='seagreen', alpha=0.8)
    axes[1, 0].set_ylabel('Energy Consumption')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_title('Mean Energy Consumption per Episode')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].bar(episodes - 0.2, data_amounts, width=0.4, label='Collected Data', color='mediumpurple', alpha=0.8)
    axes[1, 1].set_ylabel('Collected Data Amount')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_title('Collected Data Amount per Episode')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f'{prefix}episode_metrics.png'.strip('_'))
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return out_path


POLICY_DISPLAY_NAME_MAP = {
    'Our Policy (Tree)': 'GCRL-min(AoI)',
    'Our Policy': 'GCRL-min(AoI)',
    # 直接把 Our Policy (Diffusion) 在图上显示为 Dif fusion（你想要的名字）
    'Our Policy (Diffusion)': 'Dif fusion',
    'Diffusion': 'Diffusion',  # 若以后仍使用 Greedy 训练的 Diffusion，可继续复用
    'Random': 'Random',
    'Stay': 'Stay',
    'Greedy (AoI)': 'Greedy',
    'Nearest High AoI': 'Nearest High',
}


def _policy_data_4(policy_results):
    """从 policy_results 解析出 (policies, aois, energies, data_amounts, coverages)。"""
    policies = list(policy_results.keys())
    if not policies:
        return None, None, None, None, None

    def to_vec(v):
        if isinstance(v, (list, tuple)) and len(v) >= 4:
            arr = np.array(v)
            if arr.ndim == 1:
                return arr[:4]
            return np.mean(arr, axis=0)[:4]
        return np.array([0, 0, 0, 0, 0])[:4]

    data = np.array([to_vec(policy_results[p]) for p in policies])
    aois, energies, data_amounts, coverages = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    display_policies = [POLICY_DISPLAY_NAME_MAP.get(p, p) for p in policies]
    return display_policies, aois, energies, data_amounts, coverages


def plot_policy_comparison(policy_results, output_dir, fig_name='policy_comparison.png'):
    """
    多策略对比：一张图内 2x2 子图，每个指标独立纵坐标，对比更清晰。
    """
    _ensure_dir(output_dir)
    res = _policy_data_4(policy_results)
    if res[0] is None:
        return None
    policies, aois, energies, data_amounts, coverages = res

    x = np.arange(len(policies))
    width = 0.5

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('Policy Comparison (per metric)', fontsize=12)

    axes[0, 0].bar(x, aois, width, color='steelblue', alpha=0.9)
    axes[0, 0].set_ylabel('Mean AoI')
    axes[0, 0].set_title('Mean AoI (lower is better)')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(policies)
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    axes[0, 1].bar(x, coverages, width, color='coral', alpha=0.9)
    axes[0, 1].set_ylabel('User Coverage')
    axes[0, 1].set_title('User Coverage (higher is better)')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(policies)
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    axes[1, 0].bar(x, energies, width, color='seagreen', alpha=0.9)
    axes[1, 0].set_ylabel('Energy Consumption')
    axes[1, 0].set_title('Energy Consumption (higher is better)')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(policies)
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    axes[1, 1].bar(x, data_amounts, width, color='mediumpurple', alpha=0.9)
    axes[1, 1].set_ylabel('Collected Data Amount')
    axes[1, 1].set_title('Collected Data Amount (higher is better)')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(policies)
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    # 统一缩小并旋转 x 轴策略名称，避免重叠
    for ax in axes.ravel():
        for tick in ax.get_xticklabels():
            tick.set_fontsize(8)
            tick.set_rotation(20)
            tick.set_ha('right')

    plt.tight_layout()
    out_path = os.path.join(output_dir, fig_name)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return out_path


def plot_policy_comparison_separate(policy_results, output_dir):
    """
    四个指标分别生成一张对比图，每张图独立纵坐标，对比最清晰。
    生成文件：comparison_aoi.png, comparison_coverage.png, comparison_energy.png, comparison_data_amount.png
    """
    _ensure_dir(output_dir)
    res = _policy_data_4(policy_results)
    if res[0] is None:
        return []
    policies, aois, energies, data_amounts, coverages = res

    x = np.arange(len(policies))
    width = 0.5
    out_paths = []

    # 1. Mean AoI
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(x, aois, width, color='steelblue', alpha=0.9)
    ax.set_ylabel('Mean AoI')
    ax.set_title('Mean AoI (lower is better)')
    ax.set_xticks(x)
    ax.set_xticklabels(policies)
    ax.grid(True, alpha=0.3, axis='y')
    for tick in ax.get_xticklabels():
        tick.set_fontsize(8)
        tick.set_rotation(20)
        tick.set_ha('right')
    plt.tight_layout()
    p = os.path.join(output_dir, 'comparison_aoi.png')
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    out_paths.append(p)

    # 2. User Coverage
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(x, coverages, width, color='coral', alpha=0.9)
    ax.set_ylabel('User Coverage')
    ax.set_title('User Coverage (higher is better)')
    ax.set_xticks(x)
    ax.set_xticklabels(policies)
    ax.grid(True, alpha=0.3, axis='y')
    for tick in ax.get_xticklabels():
        tick.set_fontsize(8)
        tick.set_rotation(20)
        tick.set_ha('right')
    plt.tight_layout()
    p = os.path.join(output_dir, 'comparison_coverage.png')
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    out_paths.append(p)

    # 3. Energy Consumption
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(x, energies, width, color='seagreen', alpha=0.9)
    ax.set_ylabel('Energy Consumption')
    ax.set_title('Energy Consumption (higher is better)')
    ax.set_xticks(x)
    ax.set_xticklabels(policies)
    ax.grid(True, alpha=0.3, axis='y')
    for tick in ax.get_xticklabels():
        tick.set_fontsize(8)
        tick.set_rotation(20)
        tick.set_ha('right')
    plt.tight_layout()
    p = os.path.join(output_dir, 'comparison_energy.png')
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    out_paths.append(p)

    # 4. Collected Data Amount
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(x, data_amounts, width, color='mediumpurple', alpha=0.9)
    ax.set_ylabel('Collected Data Amount')
    ax.set_title('Collected Data Amount (higher is better)')
    ax.set_xticks(x)
    ax.set_xticklabels(policies)
    ax.grid(True, alpha=0.3, axis='y')
    for tick in ax.get_xticklabels():
        tick.set_fontsize(8)
        tick.set_rotation(20)
        tick.set_ha('right')
    plt.tight_layout()
    p = os.path.join(output_dir, 'comparison_data_amount.png')
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    out_paths.append(p)

    return out_paths


def plot_policy_comparison_normalized(policy_results, output_dir, fig_name='policy_comparison_normalized.png'):
    """
    多策略对比：四个指标分别做 0-1 归一化后画柱状图，便于不同量纲对比。
    归一化按列：越大越好的指标（coverage, energy, data）按列归一化；
    AoI 越小越好，取倒数后归一化使“越高越好”一致。
    """
    _ensure_dir(output_dir)
    res = _policy_data_4(policy_results)
    if res[0] is None:
        return None
    policies, aois, energies, data_amounts, coverages = res
    data = np.column_stack([aois, energies, data_amounts, coverages])
    # AoI 越小越好；coverage, energy, data 越大越好
    aoi_col = data[:, 0]
    aoi_inv = 1.0 / (aoi_col + 1e-8)  # 或用 max - aoi 做归一化
    aoi_norm = (aoi_inv - aoi_inv.min()) / (aoi_inv.max() - aoi_inv.min() + 1e-8)
    for i in range(data.shape[0]):
        data[i, 0] = aoi_norm[i]
    for j in range(1, 4):
        col = data[:, j]
        data[:, j] = (col - col.min()) / (col.max() - col.min() + 1e-8)

    x = np.arange(len(policies))
    width = 0.2
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - 1.5 * width, data[:, 0], width, label='Mean AoI (higher=better)', color='steelblue', alpha=0.9)
    ax.bar(x - 0.5 * width, data[:, 3], width, label='User Coverage', color='coral', alpha=0.9)
    ax.bar(x + 0.5 * width, data[:, 1], width, label='Energy Consumption', color='seagreen', alpha=0.9)
    ax.bar(x + 1.5 * width, data[:, 2], width, label='Collected Data Amount', color='mediumpurple', alpha=0.9)
    ax.set_ylabel('Normalized Value (0-1)')
    ax.set_title('Policy Comparison (Normalized)')
    ax.set_xticks(x)
    ax.set_xticklabels(policies)
    for tick in ax.get_xticklabels():
        tick.set_fontsize(8)
        tick.set_rotation(20)
        tick.set_ha('right')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    out_path = os.path.join(output_dir, fig_name)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return out_path


def plot_policy_comparison_radar(policy_results, output_dir, fig_name='comparison_radar.png'):
    """
    雷达图对比多策略：四轴为 AoI(取反)、Coverage、Energy、Data，同一图中多个多边形。
    """
    _ensure_dir(output_dir)
    res = _policy_data_4(policy_results)
    if res[0] is None:
        return None
    policies, aois, energies, data_amounts, coverages = res
    def _norm01(arr):
        a = np.array(arr, dtype=float)
        r = np.max(a) - np.min(a)
        if r < 1e-8:
            return np.ones_like(a) * 0.5
        return (a - np.min(a)) / r

    aoi_inv = 1.0 / (np.array(aois) + 1e-8)
    aoi_norm = _norm01(aoi_inv)
    coverages_n = _norm01(coverages)
    energies_n = _norm01(energies)
    data_n = _norm01(data_amounts)
    labels = ['AoI\n(lower→better)', 'Coverage', 'Energy', 'Data']
    num_vars = 4
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))
    colors = plt.cm.tab10(np.linspace(0, 1, len(policies)))
    for i, name in enumerate(policies):
        values = [aoi_norm[i], coverages_n[i], energies_n[i], data_n[i]]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=name, color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title('Policy Comparison (Radar, normalized)')
    plt.tight_layout()
    out_path = os.path.join(output_dir, fig_name)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return out_path


def plot_policy_comparison_with_std(policy_results_lists, output_dir, fig_name='comparison_with_std.png'):
    """
    policy_results_lists: dict, 策略名 -> list of (aoi, energy, data, coverage)。
    绘制四指标柱状图，带 mean ± std 误差条。
    """
    _ensure_dir(output_dir)
    policies = list(policy_results_lists.keys())
    if not policies:
        return None
    means, stds = [], []
    for p in policies:
        arr = np.array(policy_results_lists[p])
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        means.append(np.mean(arr, axis=0))
        stds.append(np.std(arr, axis=0) if arr.shape[0] > 1 else np.zeros(4))
    means = np.array(means)
    stds = np.array(stds)

    xtick_labels = [POLICY_DISPLAY_NAME_MAP.get(p, p) for p in policies]
    x = np.arange(len(policies))
    width = 0.5
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('Policy Comparison (mean ± std)', fontsize=12)
    axes[0, 0].bar(x, means[:, 0], width, yerr=stds[:, 0], capsize=3, color='steelblue', alpha=0.9)
    axes[0, 0].set_ylabel('Mean AoI')
    axes[0, 0].set_title('Mean AoI')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(xtick_labels)
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    axes[0, 1].bar(x, means[:, 3], width, yerr=stds[:, 3], capsize=3, color='coral', alpha=0.9)
    axes[0, 1].set_ylabel('User Coverage')
    axes[0, 1].set_title('User Coverage')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(xtick_labels)
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    axes[1, 0].bar(x, means[:, 1], width, yerr=stds[:, 1], capsize=3, color='seagreen', alpha=0.9)
    axes[1, 0].set_ylabel('Energy Consumption')
    axes[1, 0].set_title('Energy Consumption')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(xtick_labels)
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    axes[1, 1].bar(x, means[:, 2], width, yerr=stds[:, 2], capsize=3, color='mediumpurple', alpha=0.9)
    axes[1, 1].set_ylabel('Collected Data Amount')
    axes[1, 1].set_title('Collected Data Amount')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(xtick_labels)
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    # 统一缩小并旋转 x 轴策略名称，避免重叠
    for ax in axes.ravel():
        for tick in ax.get_xticklabels():
            tick.set_fontsize(8)
            tick.set_rotation(20)
            tick.set_ha('right')

    plt.tight_layout()
    out_path = os.path.join(output_dir, fig_name)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return out_path


def plot_episode_timeseries(env, output_dir, human_num, prefix='episode'):
    """
    单次 episode 内：Mean AoI 与 User Coverage 随 timestep 的变化。
    env: 需为 unwrapped 的 crowd_sim 环境，且刚跑完一个 episode。
    """
    _ensure_dir(output_dir)
    try:
        unwrapped = env.unwrapped if hasattr(env, 'unwrapped') else env
        aoi_list = getattr(unwrapped, 'mean_aoi_timelist', None)
        update_list = getattr(unwrapped, 'update_human_timelist', None)
    except Exception:
        return None
    if aoi_list is None or update_list is None or human_num <= 0:
        return None
    aoi_list = np.asarray(aoi_list).flatten()
    update_list = np.asarray(update_list).flatten()
    coverage_per_step = update_list / human_num

    fig, axes = plt.subplots(2, 1, figsize=(8, 6))
    steps_aoi = np.arange(len(aoi_list))
    axes[0].plot(steps_aoi, aoi_list, color='steelblue', alpha=0.9)
    axes[0].set_ylabel('Mean AoI')
    axes[0].set_xlabel('Timestep')
    axes[0].set_title('Mean AoI over Timestep (within one episode)')
    axes[0].grid(True, alpha=0.3)

    steps_cov = np.arange(len(update_list))
    axes[1].plot(steps_cov, coverage_per_step, color='coral', alpha=0.9)
    axes[1].set_ylabel('User Coverage (per step)')
    axes[1].set_xlabel('Timestep')
    axes[1].set_title('User Coverage over Timestep (within one episode)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f'{prefix}_timeseries.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return out_path


def plot_users_and_uav_trajectories(env, output_dir, prefix='episode'):
    """
    将 dataset 中的 mobile user 画成黑点，每个 UAV 的轨迹画成一条线。
    env: 跑完至少一个 episode 的环境（含 human_df、robot_x_timelist、robot_y_timelist）。
    """
    _ensure_dir(output_dir)
    try:
        unwrapped = env.unwrapped if hasattr(env, 'unwrapped') else env
        human_df = getattr(unwrapped, 'human_df', None)
        robot_x = getattr(unwrapped, 'robot_x_timelist', None)
        robot_y = getattr(unwrapped, 'robot_y_timelist', None)
        nlon = getattr(unwrapped, 'nlon', 200)
        nlat = getattr(unwrapped, 'nlat', 120)
        start_ts = getattr(unwrapped, 'start_timestamp', None)
        step_time = getattr(unwrapped, 'step_time', 15)
        num_timestep = getattr(unwrapped, 'num_timestep', 120)
    except Exception:
        return None
    if human_df is None or robot_x is None or robot_y is None:
        return None

    fig, ax = plt.subplots(figsize=(10, 8))

    # 只画本 episode 时间范围内的 user 位置（可选：画全量用 human_df[['x','y']]）
    if start_ts is not None and step_time is not None and 'timestamp' in human_df.columns:
        end_ts = start_ts + num_timestep * step_time
        df_ep = human_df[(human_df['timestamp'] >= start_ts) & (human_df['timestamp'] <= end_ts)]
    else:
        df_ep = human_df
    if 'x' in df_ep.columns and 'y' in df_ep.columns:
        ax.scatter(df_ep['x'], df_ep['y'], c='black', s=4, alpha=0.35, label='Mobile users')

    # 每个 UAV 一条轨迹
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']
    for i in range(robot_x.shape[1]):
        x = robot_x[:, i]
        y = robot_y[:, i]
        c = colors[i % len(colors)]
        ax.plot(x, y, color=c, linewidth=2, label=f'UAV {i+1}')
        ax.scatter(x[0], y[0], c=c, s=80, marker='o', edgecolors='black', linewidths=1.5, zorder=5)
        ax.scatter(x[-1], y[-1], c=c, s=80, marker='s', edgecolors='black', linewidths=1.5, zorder=5)
    ax.set_xlim(0, nlon)
    ax.set_ylim(0, nlat)
    ax.set_xlabel('x (grid)')
    ax.set_ylabel('y (grid)')
    ax.set_title('Mobile Users (black) & UAV Trajectories')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f'{prefix}_users_and_uav_trajectories.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return out_path


def plot_uav_energy_over_time(env, output_dir, prefix='episode'):
    """每个 UAV 的剩余能量随 timestep 变化（归一化到 0~1）。"""
    _ensure_dir(output_dir)
    try:
        unwrapped = env.unwrapped if hasattr(env, 'unwrapped') else env
        energy = getattr(unwrapped, 'robot_energy_timelist', None)
        max_energy = getattr(unwrapped, 'max_uav_energy', 1.0)
    except Exception:
        return None
    if energy is None or energy.size == 0:
        return None
    energy = np.asarray(energy)
    if energy.ndim == 1:
        energy = energy[:, None]
    T, n_uav = energy.shape[0], energy.shape[1]
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']
    for i in range(n_uav):
        normalized = energy[:, i] / max_energy
        ax.plot(np.arange(T), normalized, color=colors[i % len(colors)], label=f'UAV {i+1}', linewidth=2)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Remaining Energy (normalized)')
    ax.set_title('UAV Energy over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f'{prefix}_uav_energy.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return out_path


def plot_aoi_distribution(env, output_dir, prefix='episode'):
    """Episode 结束时各 user 的 AoI 分布直方图。"""
    _ensure_dir(output_dir)
    try:
        unwrapped = env.unwrapped if hasattr(env, 'unwrapped') else env
        aoi_list = getattr(unwrapped, 'current_human_aoi_list', None)
    except Exception:
        return None
    if aoi_list is None or len(aoi_list) == 0:
        return None
    aoi_list = np.asarray(aoi_list).flatten()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(aoi_list, bins=min(50, max(10, len(aoi_list)//5)), color='steelblue', alpha=0.8, edgecolor='black')
    ax.axvline(np.mean(aoi_list), color='coral', linestyle='--', linewidth=2, label=f'Mean = {np.mean(aoi_list):.2f}')
    ax.set_xlabel('AoI')
    ax.set_ylabel('Number of Users')
    ax.set_title('Distribution of User AoI (end of episode)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f'{prefix}_aoi_distribution.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return out_path


def plot_spatial_aoi(env, output_dir, prefix='episode'):
    """Episode 结束时 user 的空间分布，点颜色表示 AoI（高 AoI = 红，低 = 绿）。"""
    _ensure_dir(output_dir)
    try:
        unwrapped = env.unwrapped if hasattr(env, 'unwrapped') else env
        human_df = getattr(unwrapped, 'human_df', None)
        start_ts = getattr(unwrapped, 'start_timestamp', None)
        step_time = getattr(unwrapped, 'step_time', 15)
        num_timestep = getattr(unwrapped, 'num_timestep', 120)
    except Exception:
        return None
    if human_df is None or 'x' not in human_df.columns or 'y' not in human_df.columns or 'aoi' not in human_df.columns:
        return None
    end_ts = start_ts + num_timestep * step_time if start_ts is not None else human_df['timestamp'].max()
    df_end = human_df[human_df['timestamp'] == end_ts]
    if df_end.empty:
        df_end = human_df.groupby('id').last().reset_index()
    if df_end.empty or 'aoi' not in df_end.columns:
        return None
    x, y, aoi = df_end['x'].values, df_end['y'].values, df_end['aoi'].values
    aoi = np.clip(aoi, 0, None)
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(x, y, c=aoi, s=25, cmap='RdYlGn_r', alpha=0.85, vmin=0)
    plt.colorbar(sc, ax=ax, label='AoI')
    ax.set_xlabel('x (grid)')
    ax.set_ylabel('y (grid)')
    ax.set_title('User Positions colored by AoI (end of episode, red=high)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f'{prefix}_spatial_aoi.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return out_path


def plot_trajectory_colored_by_time(env, output_dir, prefix='episode'):
    """UAV 轨迹按时间着色（起点蓝→终点红），背景为 user 黑点。"""
    _ensure_dir(output_dir)
    try:
        unwrapped = env.unwrapped if hasattr(env, 'unwrapped') else env
        human_df = getattr(unwrapped, 'human_df', None)
        robot_x = getattr(unwrapped, 'robot_x_timelist', None)
        robot_y = getattr(unwrapped, 'robot_y_timelist', None)
        nlon = getattr(unwrapped, 'nlon', 200)
        nlat = getattr(unwrapped, 'nlat', 120)
        start_ts = getattr(unwrapped, 'start_timestamp', None)
        step_time = getattr(unwrapped, 'step_time', 15)
        num_timestep = getattr(unwrapped, 'num_timestep', 120)
    except Exception:
        return None
    if robot_x is None or robot_y is None:
        return None
    fig, ax = plt.subplots(figsize=(10, 8))
    if human_df is not None and 'x' in human_df.columns and start_ts is not None:
        end_ts = start_ts + num_timestep * step_time
        df_ep = human_df[(human_df['timestamp'] >= start_ts) & (human_df['timestamp'] <= end_ts)]
        ax.scatter(df_ep['x'], df_ep['y'], c='black', s=3, alpha=0.3)
    for i in range(robot_x.shape[1]):
        x, y = robot_x[:, i], robot_y[:, i]
        n = len(x)
        for seg in range(n - 1):
            ax.plot(x[seg:seg+2], y[seg:seg+2], color=plt.cm.viridis(seg / max(n-1, 1)), linewidth=2)
        ax.scatter(x[0], y[0], c='darkblue', s=100, marker='o', edgecolors='black', linewidths=1.5, zorder=5)
        ax.scatter(x[-1], y[-1], c='darkred', s=100, marker='s', edgecolors='black', linewidths=1.5, zorder=5)
    ax.set_xlim(0, nlon)
    ax.set_ylim(0, nlat)
    ax.set_xlabel('x (grid)')
    ax.set_ylabel('y (grid)')
    ax.set_title('UAV Trajectories (blue=start, red=end, color=time)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f'{prefix}_trajectory_colored_by_time.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return out_path


def plot_episode_snapshots(env, output_dir, prefix='episode', n_snapshots=4):
    """选 n_snapshots 个时刻，每时刻一子图：user 黑点 + UAV 位置。"""
    _ensure_dir(output_dir)
    try:
        unwrapped = env.unwrapped if hasattr(env, 'unwrapped') else env
        human_df = getattr(unwrapped, 'human_df', None)
        robot_x = getattr(unwrapped, 'robot_x_timelist', None)
        robot_y = getattr(unwrapped, 'robot_y_timelist', None)
        nlon = getattr(unwrapped, 'nlon', 200)
        nlat = getattr(unwrapped, 'nlat', 120)
        start_ts = getattr(unwrapped, 'start_timestamp', None)
        step_time = getattr(unwrapped, 'step_time', 15)
        num_timestep = getattr(unwrapped, 'num_timestep', 120)
    except Exception:
        return None
    if robot_x is None or robot_y is None:
        return None
    T = robot_x.shape[0]
    indices = np.linspace(0, T - 1, n_snapshots, dtype=int)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    for idx, t in enumerate(indices):
        if idx >= len(axes):
            break
        ax = axes[idx]
        ts = start_ts + t * step_time if start_ts is not None else None
        if human_df is not None and 'x' in human_df.columns and ts is not None:
            df_t = human_df[human_df['timestamp'] == ts]
            if not df_t.empty:
                ax.scatter(df_t['x'], df_t['y'], c='black', s=8, alpha=0.6)
        for i in range(robot_x.shape[1]):
            ax.scatter(robot_x[t, i], robot_y[t, i], s=120, marker='^', edgecolors='black', linewidths=1.5,
                      color=['#e74c3c', '#3498db', '#2ecc71', '#9b59b6'][i % 4], label=f'UAV {i+1}', zorder=5)
        ax.set_xlim(0, nlon)
        ax.set_ylim(0, nlat)
        ax.set_title(f't = {t}')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    plt.suptitle('Snapshots: Users (black) & UAVs (triangles)')
    plt.tight_layout()
    out_path = os.path.join(output_dir, f'{prefix}_snapshots.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return out_path
