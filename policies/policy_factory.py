from policies.model_predictive_rl import ModelPredictiveRL
from policies.random_policy import RandomPolicy
from policies.stay_policy import StayPolicy
from policies.greedy_aoi_policy import GreedyAoIPolicy
from policies.nearest_high_aoi_policy import NearestHighAoIPolicy


def none_policy():
    return None


policy_factory = dict()
policy_factory['model_predictive_rl'] = ModelPredictiveRL
policy_factory['random_policy'] = RandomPolicy
policy_factory['stay_policy'] = StayPolicy
policy_factory['greedy_aoi_policy'] = GreedyAoIPolicy
policy_factory['nearest_high_aoi_policy'] = NearestHighAoIPolicy
policy_factory['none'] = none_policy



