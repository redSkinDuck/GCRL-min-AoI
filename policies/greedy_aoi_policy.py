# -*- coding: utf-8 -*-
"""Baseline: 每步在离散动作空间里选能「覆盖到」的 AoI 总和最大的联合动作（贪心覆盖高 AoI 用户）。"""
import numpy as np
from policies.base import Policy
from envs.model.mdp import build_action_space
from configs.config import BaseEnvConfig


def compute_greedy_score(state, action):
    """
    给定 state 和 action，计算「该动作执行后能覆盖到的用户的 AoI 之和」（与 Greedy 策略的评分一致）。
    用于在 Diffusion 采样多条轨迹时，用 Greedy 标准挑选最优动作，从而在 Greedy 基础上进一步变好。
    :param state: JointState，含 robot_states / human_states（px, py 归一化 [0,1]，human 有 aoi）
    :param action: (robot_num, 2) numpy，env 尺度的 (dx, dy)
    :return: float，覆盖到的用户的 AoI 之和
    """
    cfg = BaseEnvConfig().env
    nlon, nlat = cfg.nlon, cfg.nlat
    sensing_range = cfg.sensing_range
    robot_num = cfg.robot_num
    robot_states = state.robot_states
    human_states = state.human_states
    new_pos = []
    for rid in range(robot_num):
        rx = robot_states[rid].px * nlon
        ry = robot_states[rid].py * nlat
        new_pos.append((rx + action[rid][0], ry + action[rid][1]))
    score = 0.0
    for h in human_states:
        hx = h.px * nlon
        hy = h.py * nlat
        for (nx, ny) in new_pos:
            d = np.sqrt((hx - nx) ** 2 + (hy - ny) ** 2)
            if d <= sensing_range:
                score += h.aoi
                break
    return score


class GreedyAoIPolicy(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'GreedyAoIPolicy'
        self.action_space = None
        self.config = None

    def configure(self, config, human_df):
        self.config = BaseEnvConfig()

    def set_epsilon(self, epsilon):
        pass

    def predict(self, state, current_timestep):
        if self.action_space is None:
            self.action_space = build_action_space()
        cfg = self.config.env
        nlon, nlat = cfg.nlon, cfg.nlat
        sensing_range = cfg.sensing_range
        robot_num = cfg.robot_num
        # state 里 px/py 是归一化 [0,1]，action 与 sensing_range 是 env 尺度，需反归一化
        robot_states = state.robot_states
        human_states = state.human_states

        best_score = -1
        best_action = self.action_space[0]
        for a in self.action_space:
            new_pos = []
            for rid in range(robot_num):
                rx = robot_states[rid].px * nlon
                ry = robot_states[rid].py * nlat
                new_pos.append((rx + a[rid][0], ry + a[rid][1]))
            score = 0
            for h in human_states:
                hx = h.px * nlon
                hy = h.py * nlat
                aoi = h.aoi  # 归一化或原始均可，只用于加权
                for (nx, ny) in new_pos:
                    d = np.sqrt((hx - nx) ** 2 + (hy - ny) ** 2)
                    if d <= sensing_range:
                        score += aoi
                        break
            if score > best_score:
                best_score = score
                best_action = a
        return best_action

    def get_action_trajectory(self, state, current_timestep, env, horizon):
        """
        用 Greedy 在 env 上向前滚 horizon 步，得到动作序列，用于 Diffusion BC 标注。
        调用方需在每步前保存 state，然后调用本方法会执行 env.step，所以一般每次调用会消耗 horizon 个 env 步。
        :return: (action_seq, next_state, done) 其中 action_seq 为 (H, robot_num, 2) numpy，可能不足 H 若提前 done。
        """
        if self.action_space is None:
            self.action_space = build_action_space()
        actions = []
        t = current_timestep
        for _ in range(horizon):
            a = self.predict(state, t)
            actions.append(np.array(a, dtype=np.float64))
            state, _, done, _ = env.step(a)
            t += 1
            if done:
                break
        # 不足 H 则用最后一步动作 padding
        robot_num = actions[0].shape[0] if actions else self.config.env.robot_num
        H = horizon
        action_seq = np.zeros((H, robot_num, 2), dtype=np.float64)
        for i, a in enumerate(actions):
            action_seq[i] = a
        if len(actions) < H:
            action_seq[len(actions):] = actions[-1] if actions else 0
        return action_seq, state, done
