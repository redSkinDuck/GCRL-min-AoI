# -*- coding: utf-8 -*-
"""Baseline: 每个 UAV 独立朝「当前 AoI 最高」的用户方向选一个离散动作（朝该用户移动）。"""
import numpy as np
from policies.base import Policy
from envs.model.mdp import build_action_space
from configs.config import BaseEnvConfig


class NearestHighAoIPolicy(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'NearestHighAoIPolicy'
        self.one_uav_actions = None
        self.config = None

    def configure(self, config, human_df):
        self.config = BaseEnvConfig()

    def set_epsilon(self, epsilon):
        pass

    def predict(self, state, current_timestep):
        cfg = self.config.env
        if self.one_uav_actions is None:
            self.one_uav_actions = np.array(cfg.one_uav_action_space, dtype=np.float64)
        nlon, nlat = cfg.nlon, cfg.nlat
        robot_states = state.robot_states
        human_states = state.human_states
        robot_num = cfg.robot_num
        # state 里 px/py 是归一化 [0,1]，需反归一化到 env 尺度再与 action (env 尺度) 计算
        joint = np.zeros((robot_num, 2), dtype=np.float64)
        for rid in range(robot_num):
            rx = robot_states[rid].px * nlon
            ry = robot_states[rid].py * nlat
            best_h = max(human_states, key=lambda h: h.aoi)
            tx = best_h.px * nlon
            ty = best_h.py * nlat
            dx = tx - rx
            dy = ty - ry
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < 1e-6:
                joint[rid] = [0, 0]
                continue
            # 选离散动作里与 (dx,dy) 方向最接近的（点积最大，或终点离 target 最近）
            best_idx = 0
            best_val = -1e9
            for i, (ax, ay) in enumerate(self.one_uav_actions):
                new_x = rx + ax
                new_y = ry + ay
                to_target = (tx - new_x) ** 2 + (ty - new_y) ** 2
                # 距离 target 越近越好，即 -to_target 越大越好
                if -to_target > best_val:
                    best_val = -to_target
                    best_idx = i
            joint[rid] = self.one_uav_actions[best_idx]

        # 在离散动作空间里选与 joint 最接近的联合动作
        action_space = build_action_space()
        best_idx = 0
        best_err = 1e18
        for i, a in enumerate(action_space):
            err = np.sum((a - joint) ** 2)
            if err < best_err:
                best_err = err
                best_idx = i
        return action_space[best_idx]
