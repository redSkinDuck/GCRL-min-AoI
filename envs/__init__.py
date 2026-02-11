from gym.envs.registration import register


def disable_render_order_check(env):
    """
    在 OrderEnforcing 包装上设置 disable_render_order_enforcing=True，
    允许在未先调用 reset() 的情况下调用 render()（按 gym 报错提示的做法）。
    """
    e = env
    while hasattr(e, 'env'):
        if type(e).__name__ == 'OrderEnforcing':
            e.disable_render_order_enforcing = True
            return
        e = e.env


register(
    id='CrowdSim-v0',
    entry_point='envs.crowd_sim:CrowdSim',
)
