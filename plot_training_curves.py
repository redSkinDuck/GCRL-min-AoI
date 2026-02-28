# -*- coding: utf-8 -*-
"""
从 TensorBoard 日志目录读取训练曲线并保存为 PNG。
训练时数据已写入 --output_dir（如 logs/debug），可用本脚本在训练结束后生成曲线图。

用法:
  python plot_training_curves.py --logdir logs/debug
  python plot_training_curves.py --logdir logs/debug --output_dir logs/debug/curves
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def load_tb_scalars(logdir, tag):
    """从 TensorBoard 事件目录读取某个 tag 的 (step, value) 列表。"""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:
        raise ImportError('请安装 tensorboard: pip install tensorboard')

    acc = EventAccumulator(logdir)
    acc.Reload()
    if tag not in acc.Tags().get('scalars', []):
        return None, None
    events = acc.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return np.array(steps), np.array(values)


def main():
    parser = argparse.ArgumentParser(description='从 TensorBoard 日志绘制训练曲线')
    parser.add_argument('--logdir', type=str, default='logs/debug',
                        help='TensorBoard 日志目录（即训练时的 output_dir）')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='PNG 保存目录，默认与 logdir 相同')
    args = parser.parse_args()
    output_dir = args.output_dir or args.logdir
    os.makedirs(output_dir, exist_ok=True)

    tags_train = [
        ('train/reward', 'Train Reward'),
        ('train/mean_human_aoi', 'Train Mean AoI'),
        ('train/avg user coverage', 'Train User Coverage'),
        ('train/energy_consumption (J)', 'Train Energy'),
        ('train/collected_data_amount (MB)', 'Train Collected Data'),
    ]
    tags_rl = [
        ('RL/average_v_loss', 'Value Loss'),
        ('RL/average_s_loss', 'State Predictor Loss'),
    ]

    # 1) 训练指标：2x2 + 1
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('Training Metrics', fontsize=12)
    axes = axes.flatten()
    for i, (tag, title) in enumerate(tags_train[:4]):
        steps, values = load_tb_scalars(args.logdir, tag)
        if steps is not None and len(steps) > 0:
            axes[i].plot(steps, values, color='steelblue', alpha=0.8)
        axes[i].set_title(title)
        axes[i].set_xlabel('Episode')
        axes[i].grid(True, alpha=0.3)
    plt.tight_layout()
    p1 = os.path.join(output_dir, 'training_metrics.png')
    plt.savefig(p1, dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved:', p1)

    # 2) Loss 曲线
    fig, ax = plt.subplots(figsize=(6, 4))
    for tag, label in tags_rl:
        steps, values = load_tb_scalars(args.logdir, tag)
        if steps is not None and len(steps) > 0:
            ax.plot(steps, values, label=label, alpha=0.8)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Loss')
    ax.set_title('RL Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    p2 = os.path.join(output_dir, 'training_loss.png')
    plt.savefig(p2, dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved:', p2)

    # 3) Epsilon 衰减曲线
    steps_eps, values_eps = load_tb_scalars(args.logdir, 'train/epsilon')
    if steps_eps is not None and len(steps_eps) > 0:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(steps_eps, values_eps, color='green', alpha=0.8)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Epsilon')
        ax.set_title('Exploration Rate (Epsilon) Decay')
        ax.grid(True, alpha=0.3)
        p_eps = os.path.join(output_dir, 'training_epsilon.png')
        plt.savefig(p_eps, dpi=150, bbox_inches='tight')
        plt.close()
        print('Saved:', p_eps)

    # 4) Train vs Val Reward 同图对比（当有 val 时）
    steps_train, values_train = load_tb_scalars(args.logdir, 'train/reward')
    steps_val, values_val = load_tb_scalars(args.logdir, 'val/reward')
    if steps_train is not None and len(steps_train) > 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(steps_train, values_train, color='steelblue', alpha=0.8, label='Train Reward')
        if steps_val is not None and len(steps_val) > 0:
            ax.plot(steps_val, values_val, color='coral', alpha=0.8, label='Val Reward')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Train vs Validation Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)
        p3 = os.path.join(output_dir, 'train_vs_val_reward.png')
        plt.savefig(p3, dpi=150, bbox_inches='tight')
        plt.close()
        print('Saved:', p3)
    elif steps_val is not None and len(steps_val) > 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(steps_val, values_val, color='coral', alpha=0.8, label='Val Reward')
        ax.set_xlabel('Episode (eval every N)')
        ax.set_ylabel('Val Reward')
        ax.set_title('Validation Reward')
        ax.grid(True, alpha=0.3)
        p3 = os.path.join(output_dir, 'validation_reward.png')
        plt.savefig(p3, dpi=150, bbox_inches='tight')
        plt.close()
        print('Saved:', p3)


if __name__ == '__main__':
    main()
