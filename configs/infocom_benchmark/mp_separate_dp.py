from configs.config import BaseEnvConfig, BasePolicyConfig, BaseTrainConfig, Config


class EnvConfig(BaseEnvConfig):
    def __init__(self, debug=False):
        super(EnvConfig, self).__init__(debug)


class PolicyConfig(BasePolicyConfig):
    def __init__(self, debug=False):
        super(PolicyConfig, self).__init__(debug)
        self.name = 'model_predictive_rl'

        self.model_predictive_rl = Config()
        self.model_predictive_rl.robot_state_dim = 4
        self.model_predictive_rl.human_state_dim = 4
        self.model_predictive_rl.planning_depth = 1  # 1 -> 2  shallow to dp
        self.model_predictive_rl.planning_width = 5  # 1 -> 5
        self.model_predictive_rl.do_action_clip = True  # False -> True
        self.model_predictive_rl.motion_predictor_dims = [32, 256, 256, self.model_predictive_rl.human_state_dim]
        self.model_predictive_rl.value_network_dims = [32, 256, 256, 1]
        self.model_predictive_rl.share_graph_model = False  # False!
        # Diffusion: 替代树搜索，设为 True 时用 diffusion 采样轨迹再 rollout 选最优
        self.model_predictive_rl.use_diffusion = False
        self.model_predictive_rl.diffusion_num_samples = 32  # 推理时采样条数，越大越稳但越慢
        self.model_predictive_rl.diffusion_horizon = 1
        self.model_predictive_rl.diffusion_steps = 100
        # True：将 diffusion 采样的连续动作 snap 到树使用的离散动作空间，再按 return 选最优，通常能缩小与 Tree 的差距
        self.model_predictive_rl.diffusion_discretize_output = True



class TrainConfig(BaseTrainConfig):
    def __init__(self, debug=False):
        super(TrainConfig, self).__init__(debug)
        # trainer的小trick，暂时用不上
        self.train.freeze_state_predictor = False
        self.train.detach_state_predictor = False
        self.train.reduce_sp_update_frequency = False
