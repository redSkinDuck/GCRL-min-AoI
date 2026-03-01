# -*- coding: utf-8 -*-
"""
运行 Our Policy 与 Random 策略，并生成 AoI、Coverage、能耗、数据量 的对比图。
用法示例:
  python run_comparison.py -m logs/debug --output_dir logs/comparison
  python run_comparison.py -m logs/debug --output_dir logs/comparison --n_episodes 5
"""
import argparse
import importlib.util
import logging
import os
import sys
import torch
import gym

from configs.config import set_env_dataset
from envs import disable_render_order_check
from envs.model.agent import Agent
from method.explorer import Explorer
from method.visualize_metrics import (
    plot_policy_comparison,
    plot_policy_comparison_normalized,
    plot_policy_comparison_separate,
    plot_policy_comparison_radar,
    plot_policy_comparison_with_std,
)
from policies.policy_factory import policy_factory


def set_random_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return None


def _make_args(output_dir, vis_html=False, **kwargs):
    class Args:
        pass
    args = Args()
    args.output_dir = output_dir
    args.vis_html = vis_html
    args.plot_loop = False
    args.moving_line = False
    args.phase = 'test'
    args.debug = False
    for k, v in kwargs.items():
        setattr(args, k, v)
    return args


def _setup_env_and_policy(device, config_file, model_weights=None, policy_name=None, module_name=None, dataset=None):
    """创建 env 并配置 policy，返回 (env, agent, explorer)。dataset: Purdue|NCSU|KAIST；module_name 避免多 config 互相覆盖。"""
    if dataset is not None:
        set_env_dataset(dataset)
    name = module_name if module_name else 'config'
    spec = importlib.util.spec_from_file_location(name, config_file)
    if spec is None:
        raise FileNotFoundError('Config file not found: %s' % config_file)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    env = gym.make('CrowdSim-v0')
    disable_render_order_check(env)
    human_df = env.human_df

    policy_config = config.PolicyConfig(False)
    policy = policy_factory[policy_config.name]()
    policy.set_device(device)
    policy.configure(policy_config, human_df)
    if getattr(policy, 'trainable', False) and model_weights and os.path.isfile(model_weights):
        policy.load_model(model_weights)
    policy.set_phase('test')
    policy.set_env(env)

    agent = Agent()
    agent.set_policy(policy)
    env.set_agent(agent)
    explorer = Explorer(env, agent, device, None, gamma=0.9)
    return env, agent, explorer


def main():
    parser = argparse.ArgumentParser(description='Our Policy vs Random 对比并画图')
    parser.add_argument('-m', '--model_dir', type=str, default='logs/debug',
                        help='Our policy 模型目录（含 config.py 与 best_val.pth）')
    parser.add_argument('--output_dir', type=str, default='logs/comparison',
                        help='对比图保存目录')
    parser.add_argument('--n_episodes', type=int, default=5,
                        help='每种策略运行的 episode 数（用于取平均）')
    parser.add_argument('--random_config', type=str, default='configs/infocom_benchmark/random.py',
                        help='Random 策略的 config')
    parser.add_argument('--gpu', action='store_true', help='使用 GPU')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--compare_diffusion', action='store_true',
                        help='额外跑 Our Policy (Diffusion)，生成 Tree / Diffusion / Random 三路对比图')
    parser.add_argument('--compare_greedy_diffusion', action='store_true',
                        help='额外跑 Greedy 训练的 Diffusion 策略，在图里显示为 Diffusion')
    parser.add_argument('--greedy_diffusion_ckpt', type=str, default=None,
                        help='Greedy-Diffusion 权重路径，默认 model_dir/greedy_diffusion/best_val_greedy_diffusion.pth')
    parser.add_argument('--baselines', type=str, nargs='*', default=['stay', 'greedy_aoi', 'nearest_high_aoi'],
                        help='额外 rule-based baseline：stay, greedy_aoi, nearest_high_aoi；空则不加')
    parser.add_argument('--dataset', type=str, default='Purdue',
                        help='数据集：Purdue, NCSU, KAIST')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    os.makedirs(args.output_dir, exist_ok=True)
    set_random_seeds(args.seed)
    device = torch.device('cuda:0' if args.gpu and torch.cuda.is_available() else 'cpu')
    logging.info('Dataset: %s', args.dataset)

    run_args = _make_args(args.output_dir, vis_html=False)

    # statistics 为 6 元组: (reward, avg_return, mean_aoi, mean_energy, collected_data, coverage)
    def to_4tuple(stats):
        if stats is None:
            return (0.0, 0.0, 0.0, 0.0)
        return (float(stats[2]), float(stats[3]), float(stats[4]), float(stats[5]))

    policy_results = {}
    policy_results_lists = {}

    # 1) Our Policy：每轮 k=1 跑 n_episodes 次，收集列表用于 mean±std
    config_our = os.path.join(args.model_dir, 'config.py')
    weights_our = os.path.join(args.model_dir, 'best_val.pth')
    if not os.path.isfile(config_our):
        logging.warning('Our policy config 不存在: %s，跳过 Our Policy', config_our)
    else:
        logging.info('Running Our Policy - Tree (%d episodes)...', args.n_episodes)
        env_our, agent_our, explorer_our = _setup_env_and_policy(
            device, config_our, model_weights=weights_our, dataset=args.dataset)
        list_our = []
        for _ in range(args.n_episodes):
            explorer_our.run_k_episodes(k=1, phase='test', args=run_args, plot_index=0)
            if explorer_our.statistics is not None:
                list_our.append(to_4tuple(explorer_our.statistics))
        if list_our:
            name_tree = 'Our Policy (Tree)' if args.compare_diffusion else 'Our Policy'
            policy_results_lists[name_tree] = list_our
            policy_results[name_tree] = (
                sum(t[0] for t in list_our) / len(list_our),
                sum(t[1] for t in list_our) / len(list_our),
                sum(t[2] for t in list_our) / len(list_our),
                sum(t[3] for t in list_our) / len(list_our),
            )
            logging.info('%s stats (mean): %s', name_tree, policy_results.get(name_tree))

        if args.compare_diffusion:
            logging.info('Running Our Policy - Diffusion (%d episodes)...', args.n_episodes)
            env_diff, agent_diff, explorer_diff = _setup_env_and_policy(
                device, config_our, model_weights=weights_our, dataset=args.dataset)
            policy_diff = getattr(explorer_diff.robot, 'policy', None)
            if policy_diff is not None and hasattr(policy_diff, 'enable_diffusion'):
                policy_diff.enable_diffusion()
                weights_diff = os.path.join(args.model_dir, 'best_val_with_diffusion.pth')
                load_file = weights_diff if os.path.isfile(weights_diff) else weights_our
                if os.path.isfile(load_file):
                    try:
                        ckpt = torch.load(load_file, map_location=device)
                        if isinstance(ckpt, dict) and 'diffusion_model' in ckpt:
                            policy_diff.diffusion_model.load_state_dict(ckpt['diffusion_model'])
                            logging.info('Loaded diffusion weights from %s', load_file)
                    except Exception as e:
                        logging.warning('Could not load diffusion weights: %s', e)
            list_diff = []
            for _ in range(args.n_episodes):
                explorer_diff.run_k_episodes(k=1, phase='test', args=run_args, plot_index=0)
                if explorer_diff.statistics is not None:
                    list_diff.append(to_4tuple(explorer_diff.statistics))
            if list_diff:
                policy_results_lists['Our Policy (Diffusion)'] = list_diff
                policy_results['Our Policy (Diffusion)'] = (
                    sum(t[0] for t in list_diff) / len(list_diff),
                    sum(t[1] for t in list_diff) / len(list_diff),
                    sum(t[2] for t in list_diff) / len(list_diff),
                    sum(t[3] for t in list_diff) / len(list_diff),
                )
                logging.info('Our Policy (Diffusion) stats (mean): %s', policy_results.get('Our Policy (Diffusion)'))

        # Greedy 训练的 Diffusion（图里显示为 Diffusion）：加 --compare_greedy_diffusion 或默认路径存在权重时自动加入
        ckpt_path = args.greedy_diffusion_ckpt or os.path.join(
            args.model_dir, 'greedy_diffusion', 'best_val_greedy_diffusion.pth')
        if args.compare_greedy_diffusion or os.path.isfile(ckpt_path):
            if not os.path.isfile(ckpt_path):
                logging.warning('Greedy-Diffusion 权重不存在: %s，跳过 Diffusion', ckpt_path)
            else:
                logging.info('Running Diffusion (Greedy-trained) (%d episodes)...', args.n_episodes)
                env_gd, agent_gd, explorer_gd = _setup_env_and_policy(
                    device, config_our, model_weights=weights_our, dataset=args.dataset)
                policy_gd = getattr(explorer_gd.robot, 'policy', None)
                if policy_gd is not None and hasattr(policy_gd, 'enable_diffusion'):
                    ckpt = {}
                    try:
                        ckpt = torch.load(ckpt_path, map_location=device)
                        # 多步 checkpoint 含 diffusion_horizon，须在 enable_diffusion 前设置以创建正确结构的 diffusion
                        if isinstance(ckpt, dict) and 'diffusion_horizon' in ckpt:
                            policy_gd.diffusion_horizon = ckpt['diffusion_horizon']
                            logging.info('Greedy-Diffusion horizon=%d (multi-step)', ckpt['diffusion_horizon'])
                    except Exception as e:
                        logging.warning('Load Greedy-Diffusion checkpoint for horizon failed: %s', e)
                    policy_gd.enable_diffusion()
                    if hasattr(policy_gd, 'diffusion_discretize_output'):
                        policy_gd.diffusion_discretize_output = True
                    if hasattr(policy_gd, 'use_greedy_score_for_diffusion'):
                        policy_gd.use_greedy_score_for_diffusion = True
                    try:
                        if isinstance(ckpt, dict) and 'diffusion_model' in ckpt:
                            policy_gd.diffusion_model.load_state_dict(ckpt['diffusion_model'])
                            logging.info('Loaded Greedy-Diffusion from %s', ckpt_path)
                    except Exception as e:
                        logging.warning('Load Greedy-Diffusion weights failed: %s', e)
                list_gd = []
                for _ in range(args.n_episodes):
                    explorer_gd.run_k_episodes(k=1, phase='test', args=run_args, plot_index=0)
                    if explorer_gd.statistics is not None:
                        list_gd.append(to_4tuple(explorer_gd.statistics))
                if list_gd:
                    policy_results_lists['Diffusion'] = list_gd
                    policy_results['Diffusion'] = (
                        sum(t[0] for t in list_gd) / len(list_gd),
                        sum(t[1] for t in list_gd) / len(list_gd),
                        sum(t[2] for t in list_gd) / len(list_gd),
                        sum(t[3] for t in list_gd) / len(list_gd),
                    )
                    logging.info('Diffusion stats (mean): %s', policy_results.get('Diffusion'))

    # 2) Random
    if not os.path.isfile(args.random_config):
        logging.warning('Random config 不存在: %s，跳过 Random', args.random_config)
    else:
        logging.info('Running Random Policy (%d episodes)...', args.n_episodes)
        env_rand, agent_rand, explorer_rand = _setup_env_and_policy(device, args.random_config, dataset=args.dataset)
        list_rand = []
        for _ in range(args.n_episodes):
            explorer_rand.run_k_episodes(k=1, phase='test', args=run_args, plot_index=0)
            if explorer_rand.statistics is not None:
                list_rand.append(to_4tuple(explorer_rand.statistics))
        if list_rand:
            policy_results_lists['Random'] = list_rand
            policy_results['Random'] = (
                sum(t[0] for t in list_rand) / len(list_rand),
                sum(t[1] for t in list_rand) / len(list_rand),
                sum(t[2] for t in list_rand) / len(list_rand),
                sum(t[3] for t in list_rand) / len(list_rand),
            )
        logging.info('Random Policy stats (mean): %s', policy_results.get('Random'))

    # 3) 其他 baseline：Stay, Greedy (AoI), Nearest High AoI
    baseline_configs = {
        'stay': ('configs/infocom_benchmark/stay.py', 'Stay'),
        'greedy_aoi': ('configs/infocom_benchmark/greedy_aoi.py', 'Greedy (AoI)'),
        'nearest_high_aoi': ('configs/infocom_benchmark/nearest_high_aoi.py', 'Nearest High AoI'),
    }
    for bl in args.baselines:
        if bl not in baseline_configs:
            logging.warning('Unknown baseline: %s，跳过', bl)
            continue
        config_path, display_name = baseline_configs[bl]
        if not os.path.isfile(config_path):
            logging.warning('Baseline config 不存在: %s，跳过 %s', config_path, display_name)
            continue
        logging.info('Running %s (%d episodes)...', display_name, args.n_episodes)
        env_bl, agent_bl, explorer_bl = _setup_env_and_policy(
            device, config_path, module_name='config_baseline_%s' % bl, dataset=args.dataset)
        list_bl = []
        for _ in range(args.n_episodes):
            explorer_bl.run_k_episodes(k=1, phase='test', args=run_args, plot_index=0)
            if explorer_bl.statistics is not None:
                list_bl.append(to_4tuple(explorer_bl.statistics))
        if list_bl:
            policy_results_lists[display_name] = list_bl
            policy_results[display_name] = (
                sum(t[0] for t in list_bl) / len(list_bl),
                sum(t[1] for t in list_bl) / len(list_bl),
                sum(t[2] for t in list_bl) / len(list_bl),
                sum(t[3] for t in list_bl) / len(list_bl),
            )
            logging.info('%s stats (mean): %s', display_name, policy_results.get(display_name))

    if len(policy_results) < 2:
        logging.warning('至少需要两种策略才能画对比图，当前仅: %s', list(policy_results.keys()))
        return

    # 画对比图：2x2 汇总 + 归一化 + 分指标四张 + 雷达图 + 带误差条
    p1 = plot_policy_comparison(policy_results, args.output_dir)
    p2 = plot_policy_comparison_normalized(policy_results, args.output_dir)
    paths_separate = plot_policy_comparison_separate(policy_results, args.output_dir)
    p_radar = plot_policy_comparison_radar(policy_results, args.output_dir)
    p_std = plot_policy_comparison_with_std(policy_results_lists, args.output_dir)
    logging.info('对比图已保存: %s, %s, 雷达图: %s, 误差条: %s', p1, p2, p_radar, p_std)
    logging.info('分指标对比图: %s', paths_separate)


if __name__ == '__main__':
    main()
