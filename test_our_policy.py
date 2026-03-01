import logging
import argparse
import importlib.util
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import gym
from method.explorer import Explorer
from method.visualize_metrics import (
    plot_episode_metrics,
    plot_episode_timeseries,
    plot_users_and_uav_trajectories,
    plot_uav_energy_over_time,
    plot_aoi_distribution,
    plot_spatial_aoi,
    plot_trajectory_colored_by_time,
    plot_episode_snapshots,
)
from policies.policy_factory import policy_factory
from envs.model.agent import Agent
from configs.config import set_env_dataset


def set_random_seeds(seed):
    """
    Sets the random seeds for pytorch cpu and gpu
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return None


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    set_random_seeds(args.randomseed)
    args.output_dir = getattr(args, 'output_dir', None) or args.model_dir
    if getattr(args, 'dataset', None):
        set_env_dataset(args.dataset)
        logging.info('Using dataset: %s', args.dataset)

    # configure logging and device
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)

    if args.config is not None:
        config_file = args.config
    else:
        config_file = os.path.join(args.model_dir, 'config.py')  # TODO：注意这里需要改
    best_val_path = os.path.join(args.model_dir, 'best_val.pth')
    rl_model_path = os.path.join(args.model_dir, 'rl_model.pth')
    if os.path.isfile(best_val_path):
        model_weights = best_val_path
        logging.info('Loaded RL weights with best VAL: %s', best_val_path)
    elif os.path.isfile(rl_model_path):
        model_weights = rl_model_path
        logging.warning('best_val.pth 不存在，使用 rl_model.pth。请先完整训练以得到 best_val.pth。')
    else:
        raise FileNotFoundError(
            '未找到模型权重。请先运行训练：\n'
            '  python train_our_policy.py --config configs/infocom_benchmark/mp_separate_dp.py --output_dir logs/debug --overwrite\n'
            '或确认 -m/--model_dir 指向的目录下存在 best_val.pth 或 rl_model.pth。'
        )

    spec = importlib.util.spec_from_file_location('config', config_file)
    if spec is None:
        parser.error('Config file not found.')
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    # configure environment
    env = gym.make('CrowdSim-v0')
    from envs import disable_render_order_check
    disable_render_order_check(env)
    agent = Agent()
    human_df = env.human_df

    # configure policy
    policy_config = config.PolicyConfig(args.debug)
    policy = policy_factory[policy_config.name]()
    if args.planning_depth is not None:
        policy_config.model_predictive_rl.do_action_clip = True
        policy_config.model_predictive_rl.planning_depth = args.planning_depth
    if args.planning_width is not None:
        policy_config.model_predictive_rl.do_action_clip = True
        policy_config.model_predictive_rl.planning_width = args.planning_width
    if args.sparse_search:
        policy_config.model_predictive_rl.sparse_search = True

    policy.set_device(device)
    policy.configure(policy_config, human_df)
    if policy.trainable:
        if args.model_dir is None:
            parser.error('Trainable policy must be specified with a model weights directory')
        policy.load_model(model_weights)

    policy.set_phase(args.phase)
    policy.set_env(env)
    agent.set_policy(policy)
    agent.print_info()
    env.set_agent(agent)

    explorer = Explorer(env, agent, device, None, gamma=0.9)

    all_episode_stats = []
    for i in range(10):
        explorer.run_k_episodes(k=1, phase=args.phase, args=args, plot_index=i+1)
        if explorer.statistics is not None:
            all_episode_stats.append(explorer.statistics)
        logging.info(f'Testing #{i} finished!')

    # 多轮指标对比图：AoI、Coverage、能耗、数据量随 episode 变化
    if all_episode_stats:
        out_path = plot_episode_metrics(all_episode_stats, args.output_dir, prefix='our_policy_')
        logging.info('Episode metrics plot saved to: %s', out_path)

    # 单 episode 内 AoI / Coverage 随 timestep 变化（取最后一轮的 env 状态）
    try:
        human_num = getattr(env.unwrapped, 'human_num', 0)
        if human_num > 0:
            ts_path = plot_episode_timeseries(env, args.output_dir, human_num, prefix='our_policy')
            if ts_path:
                logging.info('Episode timeseries plot saved to: %s', ts_path)
    except Exception as e:
        logging.debug('Skip episode timeseries: %s', e)

    # 黑点 = mobile users，线条 = 每个 UAV 轨迹（取最后一轮）
    try:
        traj_path = plot_users_and_uav_trajectories(env, args.output_dir, prefix='our_policy')
        if traj_path:
            logging.info('Users & UAV trajectories plot saved to: %s', traj_path)
    except Exception as e:
        logging.debug('Skip users & UAV trajectories: %s', e)

    # 更多可视化：能量曲线、AoI 分布、空间 AoI、按时间着色的轨迹、时刻快照
    for name, plot_fn in [
        ('UAV energy', plot_uav_energy_over_time),
        ('AoI distribution', plot_aoi_distribution),
        ('Spatial AoI', plot_spatial_aoi),
        ('Trajectory colored by time', plot_trajectory_colored_by_time),
        ('Episode snapshots', plot_episode_snapshots),
    ]:
        try:
            path = plot_fn(env, args.output_dir, prefix='our_policy')
            if path:
                logging.info('%s saved to: %s', name, path)
        except Exception as e:
            logging.debug('Skip %s: %s', name, e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('-m', '--model_dir', type=str, default="logs/debug")
    parser.add_argument('--output_dir', type=str, default=None,
                        help='保存可视化结果的目录，默认与 model_dir 相同')
    parser.add_argument('--dataset', type=str, default=None,
                        help='评估数据集：Purdue, NCSU, KAIST；不指定则使用 config 默认')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--gpu_id', type=str, default='1')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--randomseed', type=int, default=0)

    parser.add_argument('--vis_html', default=False, action='store_true')
    parser.add_argument('--plot_loop', default=False, action='store_true')

    parser.add_argument('-d', '--planning_depth', type=int, default=None)
    parser.add_argument('-w', '--planning_width', type=int, default=None)
    parser.add_argument('--sparse_search', default=False, action='store_true')
    
    parser.add_argument('--moving_line', default=False, action='store_true')

    sys_args = parser.parse_args()

    main(sys_args)
