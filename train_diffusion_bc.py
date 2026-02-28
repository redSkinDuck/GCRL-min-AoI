# -*- coding: utf-8 -*-
"""
用树搜索策略收集 (state, action_sequence) 做行为克隆，训练 Diffusion 模型。
支持 DAgger：多轮用「当前 Diffusion + Tree 混合」跑环境，每步用 Tree 标 label，减轻 OOD。

用法:
  python train_diffusion_bc.py -m logs/debug --output_dir logs/debug --steps 8000 --epochs 150
  python train_diffusion_bc.py -m logs/debug --dagger_rounds 3 --steps 5000  # DAgger 3 轮
"""
import argparse
import importlib.util
import logging
import os
import random
import torch
import gym
from torch.utils.data import DataLoader, TensorDataset

from envs import disable_render_order_check
from envs.model.agent import Agent
from policies.policy_factory import policy_factory
from configs.config import BaseEnvConfig

tmp_config = BaseEnvConfig()
ACTION_SCALE = 30.0


def collect_with_policy(env, policy, device, steps, use_diffusion_prob=0.0, log_every=500):
    """用 tree 的 get_action_trajectory 标 label；用 (1-ε)*tree + ε*diffusion 控制 env 步进。
    policy 需同时支持 get_action_trajectory（tree）和 predict（可切 use_diffusion）。"""
    agent = Agent()
    agent.set_policy(policy)
    policy.set_phase('test')
    env.set_agent(agent)
    states_robot, states_human, action_seqs = [], [], []
    step_count = 0
    while step_count < steps:
        state = env.reset(phase='test')
        done = False
        t = 0
        while not done and step_count < steps:
            # 始终用 tree 轨迹作 label（与 use_diffusion 无关）
            action_seq = policy.get_action_trajectory(state, t)
            state_tensor = state.to_tensor(add_batch_size=True, device=device)
            states_robot.append(state_tensor[0].cpu().squeeze(0))
            states_human.append(state_tensor[1].cpu().squeeze(0))
            action_seqs.append(torch.tensor(action_seq / ACTION_SCALE, dtype=torch.float32).clamp(-1, 1))
            step_count += 1
            if step_count % log_every == 0:
                logging.info('Collecting: %d / %d steps', step_count, steps)
            if action_seq.shape[0] < 1:
                break
            # 用 tree 或 diffusion 选动作 step（DAgger 时部分用 diffusion 以访问其会到的状态）
            if use_diffusion_prob > 0 and random.random() < use_diffusion_prob and getattr(policy, 'diffusion_model', None) is not None:
                policy.use_diffusion = True
                action = policy.predict(state, t)
            else:
                action = action_seq[0]
            state, _, done, _ = env.step(action)
            t += 1
    return states_robot, states_human, action_seqs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_dir', type=str, default='logs/debug')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--steps', type=int, default=8000, help='每轮收集的 (s,a_seq) 条数')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=150, help='每轮 BC 训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dagger_rounds', type=int, default=1, help='DAgger 轮数；1=只做一轮 BC，>1 时后续轮用 diffusion 混合 rollout')
    parser.add_argument('--dagger_epsilon', type=float, default=0.5, help='DAgger 第2轮起用 diffusion 做 rollout 的概率')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    args.output_dir = args.output_dir or args.model_dir

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    torch.manual_seed(args.seed)
    random.seed(args.seed)

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
    policy_tree = policy_factory[policy_config.name]()
    policy_tree.set_device(device)
    policy_tree.configure(policy_config, env.human_df)
    if os.path.isfile(weights_file):
        policy_tree.load_model(weights_file)
    policy_tree.set_env(env)

    all_robot, all_human, all_action = [], [], []

    for round_idx in range(args.dagger_rounds):
        logging.info('========== DAgger round %d / %d ==========', round_idx + 1, args.dagger_rounds)
        use_diffusion_prob = args.dagger_epsilon if round_idx > 0 else 0.0
        rb, rh, ac = collect_with_policy(
            env, policy_tree, device, args.steps, use_diffusion_prob=use_diffusion_prob)
        all_robot.extend(rb)
        all_human.extend(rh)
        all_action.extend(ac)
        logging.info('Collected %d new samples, total %d', len(rb), len(all_robot))

        if len(all_robot) < args.batch_size:
            logging.warning('Total samples %d < batch_size %d', len(all_robot), args.batch_size)
            continue

        H = min(a.size(0) for a in all_action)
        robot_num = all_action[0].size(1)
        action_dim = all_action[0].size(2)
        action_flat = [a[:H].flatten() for a in all_action]

        dataset = TensorDataset(
            torch.stack(all_robot),
            torch.stack(all_human),
            torch.stack(action_flat),
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        if round_idx == 0:
            policy_tree.enable_diffusion()
        diff = policy_tree.diffusion_model
        opt = torch.optim.Adam(diff.parameters(), lr=args.lr)
        graph = policy_tree.value_estimator.graph_model

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
            logging.info('Round %d Epoch %d loss %.4f', round_idx + 1, epoch + 1, total_loss / max(n_batch, 1))

    ckpt = torch.load(weights_file, map_location=device) if os.path.isfile(weights_file) else {}
    if isinstance(ckpt, dict):
        ckpt['diffusion_model'] = policy_tree.diffusion_model.state_dict()
    else:
        ckpt = {'diffusion_model': policy_tree.diffusion_model.state_dict()}
    out_path = os.path.join(args.output_dir, 'best_val_with_diffusion.pth')
    torch.save(ckpt, out_path)
    logging.info('Saved checkpoint with diffusion to %s', out_path)


if __name__ == '__main__':
    main()
