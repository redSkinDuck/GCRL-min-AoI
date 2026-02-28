# -*- coding: utf-8 -*-
"""Baseline: UAV 不移动，始终选择 (0,0)。"""
import numpy as np
from policies.base import Policy
from envs.model.mdp import build_action_space


class StayPolicy(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'StayPolicy'
        self.action_space = None

    def configure(self, config, human_df):
        return

    def set_epsilon(self, epsilon):
        pass

    def predict(self, state, current_timestep):
        if self.action_space is None:
            self.action_space = build_action_space()
        # 选所有 UAV 都不动的联合动作：每个 UAV (0,0)
        for a in self.action_space:
            if np.allclose(a, 0):
                return a
        return self.action_space[0]
