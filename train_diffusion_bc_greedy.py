# -*- coding: utf-8 -*-
"""
用 Greedy (AoI) 策略在 env 上 rollout 得到 (state, action_sequence)，对 Diffusion 做 BC。
Diffusion 结构和 state embedding 复用 model_dir 里的 RL 配置与 graph_model，只训练 diffusion 权重。

用法:
  python train_diffusion_bc_greedy.py -m logs/debug --greedy_config configs/infocom_benchmark/greedy_aoi.py --output_dir logs/greedy_diffusion --horizon 1 --steps 8000
  python train_diffusion_bc_greedy.py -m logs/debug --greedy_config configs/infocom_benchmark/greedy_aoi.py --horizon 5 --steps 5000 --epochs 100
"""
import argparse
import importlib.util
import logging
import os
import torch
import gym
from torch.utils.data import DataLoader, TensorDataset

from envs import disable_render_order_check
from envs.model.agent import Agent
from policies.policy_factory import policy_factory
from configs.config import BaseEnvConfig

tmp_config = BaseEnvConfig()
ACTION_SCALE = 30.0


def collect_with_greedy(env, greedy_policy, device, steps, horizon, log_every=500):
    """
    用 Greedy 跑 env，每 horizon 步存一条 (state_t, action_sequence)。
    会执行 env.step，所以每存一条样本会消耗 horizon 个 env 步。
    """
    agent = Agent()
    agent.set_policy(greedy_policy)
    greedy_policy.set_phase('test')
    env.set_agent(agent)

    states_robot, states_human, action_seqs = [], [], []
    step_count = 0
    while step_count < steps:
        state = env.reset(phase='test')
        done = False
        t = 0
        while not done and step_count < steps:
            state_t = state
            action_seq, state, done = greedy_policy.get_action_trajectory(state, t, env, horizon)
            state_tensor = state_t.to_tensor(add_batch_size=True, device=device)
            states_robot.append(state_tensor[0].cpu().squeeze(0))
            states_human.append(state_tensor[1].cpu().squeeze(0))
            action_seqs.append(
                torch.tensor(action_seq / ACTION_SCALE, dtype=torch.float32).clamp(-1.0, 1.0)
            )
            step_count += 1
            t += horizon
            if step_count % log_every == 0:
                logging.info('Collecting: %d / %d samples (Greedy, H=%d)', step_count, steps, horizon)
    return states_robot, states_human, action_seqs


def main():
    parser = argparse.ArgumentParser(description='Train Diffusion with BC from Greedy trajectories')
    parser.add_argument('-m', '--model_dir', type=str, default='logs/debug',
                        help='RL 模型目录，用于 graph_model 与 diffusion 结构')
    parser.add_argument('--greedy_config', type=str, default='configs/infocom_benchmark/greedy_aoi.py',
                        help='Greedy 策略的 config 路径')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--steps', type=int, default=8000, help='收集的 (s, a_seq) 条数')
    parser.add_argument('--horizon', type=int, default=3, help='轨迹长度 H，>1 时做多步前瞻，有机会超过单步 Greedy')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    args.output_dir = args.output_dir or os.path.join(args.model_dir, 'greedy_diffusion')
    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    torch.manual_seed(args.seed)

    # 1) 加载 RL 配置与权重（只为了 graph_model + diffusion 结构）
    config_file = os.path.join(args.model_dir, 'config.py')
    weights_file = os.path.join(args.model_dir, 'best_val.pth')
    if not os.path.isfile(config_file):
        raise FileNotFoundError('Config not found: %s' % config_file)
    spec = importlib.util.spec_from_file_location('config', config_file)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = gym.make('CrowdSim-v0')
    disable_render_order_check(env)

    policy_config = config.PolicyConfig(False)
    policy_config.model_predictive_rl.use_diffusion = False
    policy_rl = policy_factory[policy_config.name]()
    policy_rl.set_device(device)
    policy_rl.configure(policy_config, env.human_df)
    if os.path.isfile(weights_file):
        policy_rl.load_model(weights_file)
    policy_rl.set_env(env)
    # 用 --horizon 创建 diffusion，多步轨迹才能做多步 Greedy 打分、超越单步 Greedy
    policy_rl.diffusion_horizon = args.horizon
    policy_rl.enable_diffusion()
    # diffusion 结构已创建，下面用 Greedy 数据训练，不加载 diffusion 权重

    # 2) 加载 Greedy 策略
    if not os.path.isfile(args.greedy_config):
        raise FileNotFoundError('Greedy config not found: %s' % args.greedy_config)
    spec_g = importlib.util.spec_from_file_location('greedy_config', args.greedy_config)
    config_g = importlib.util.module_from_spec(spec_g)
    spec_g.loader.exec_module(config_g)
    policy_greedy = policy_factory[config_g.PolicyConfig(False).name]()
    policy_greedy.configure(config_g.PolicyConfig(False), env.human_df)
    policy_greedy.set_env(env)

    # 3) 用 Greedy 收集 (state, action_sequence)
    logging.info('Collecting %d samples with Greedy (horizon=%d)...', args.steps, args.horizon)
    states_robot, states_human, action_seqs = collect_with_greedy(
        env, policy_greedy, device, args.steps, args.horizon
    )

    if len(states_robot) < args.batch_size:
        raise ValueError('Samples %d < batch_size %d' % (len(states_robot), args.batch_size))

    H = min(a.size(0) for a in action_seqs)
    robot_num = action_seqs[0].size(1)
    action_dim = action_seqs[0].size(2)
    action_flat = [a[:H].flatten() for a in action_seqs]

    dataset = TensorDataset(
        torch.stack(states_robot),
        torch.stack(states_human),
        torch.stack(action_flat),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 4) 训练 Diffusion（BC）
    diff = policy_rl.diffusion_model
    graph = policy_rl.value_estimator.graph_model
    opt = torch.optim.Adam(diff.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        total_loss = 0.0
        n_batch = 0
        for rb, rh, ac in loader:
            rb, rh, ac = rb.to(device), rh.to(device), ac.to(device)
            ac = ac.view(ac.size(0), H, robot_num, action_dim)
            loss = diff((rb, rh), graph, ac)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batch += 1
        logging.info('Epoch %d loss %.4f', epoch + 1, total_loss / max(n_batch, 1))

    # 5) 保存：与 RL 的 best_val.pth 合并，得到带 Greedy-Diffusion 的 checkpoint（含 diffusion_horizon 供推理用）
    ckpt = torch.load(weights_file, map_location=device) if os.path.isfile(weights_file) else {}
    if isinstance(ckpt, dict):
        ckpt['diffusion_model'] = policy_rl.diffusion_model.state_dict()
        ckpt['diffusion_horizon'] = args.horizon
    else:
        ckpt = {'diffusion_model': policy_rl.diffusion_model.state_dict(), 'diffusion_horizon': args.horizon}
    out_path = os.path.join(args.output_dir, 'best_val_greedy_diffusion.pth')
    torch.save(ckpt, out_path)
    logging.info('Saved Greedy-Diffusion checkpoint to %s (horizon=%d)', out_path, args.horizon)


if __name__ == '__main__':
    main()
