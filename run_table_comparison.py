# -*- coding: utf-8 -*-
"""
生成论文风格的两张表：Random / Tree / Diffusion 在 Purdue、NCSU、KAIST 上的性能与耗时。
- TABLE I: Impact of different policies (R=Mean AoI, ψ=Coverage, ρ=Energy, ς=Data)
- TABLE II: Computational complexity by time cost (ms, 每步决策平均耗时)

表中显示名：Tree -> GCRL-min(AoI), Diffusion -> Diffusion-based (AoI)
支持多种子 --seeds 报 mean±std；支持 ablation --planning_depths/--planning_widths

用法:
  python run_table_comparison.py -m logs/debug --output_dir logs/comparison --compare_diffusion
  python run_table_comparison.py -m logs/debug --seeds 0 1 2 3 4 --compare_diffusion   # variance
  python run_table_comparison.py -m logs/debug --planning_depths 1 2 --planning_widths 5  # ablation
"""
import argparse
import importlib.util
import logging
import os
import numpy as np
import torch
import gym

from envs import disable_render_order_check
from envs.model.agent import Agent
from method.explorer import Explorer
from policies.policy_factory import policy_factory
from configs.config import set_env_dataset

# 表中显示名（仅改表和图，内部仍用 Tree/Diffusion/Random 等 key）
DISPLAY_NAMES = {
    'Random': 'Random',
    'Tree': 'GCRL-min(AoI)',
    'Diffusion': 'Diffusion-based (AoI)',
    'Stay': 'Stay',
    'GreedyAoI': 'Greedy (AoI)',
    'NearestHighAoI': 'Nearest High AoI',
}


def set_random_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _make_args(output_dir, **kwargs):
    class Args:
        pass
    args = Args()
    args.output_dir = output_dir
    args.vis_html = False
    args.plot_loop = False
    args.moving_line = False
    args.phase = 'test'
    args.debug = False
    for k, v in kwargs.items():
        setattr(args, k, v)
    return args


def _display_name(method_key):
    """表中显示名：Tree -> GCRL-min(AoI)，Diffusion -> Diffusion-based (AoI)，Tree_d1_w5 -> GCRL-min(AoI) (d=1,w=5)。"""
    if method_key in DISPLAY_NAMES:
        return DISPLAY_NAMES[method_key]
    if method_key.startswith('Tree_'):
        parts = method_key.replace('Tree_', '').split('_')
        suffix = ', '.join(p.replace('d', 'd=', 1).replace('w', 'w=', 1) for p in parts if p)
        return f'GCRL-min(AoI) ({suffix})' if suffix else DISPLAY_NAMES['Tree']
    return method_key


def _setup_env_and_policy(device, config_file, model_weights=None, policy_name=None, dataset=None,
                          planning_depth=None, planning_width=None, module_name=None):
    """创建 env 并配置 policy。若 dataset 不为 None，先 set_env_dataset(dataset)。
    planning_depth/planning_width 用于 ablation：覆盖 config 中的值。
    module_name: 加载 config 时用的模块名，避免多 config 互相覆盖（baseline 必传唯一名）。"""
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
    if planning_depth is not None:
        policy_config.model_predictive_rl.planning_depth = planning_depth
        policy_config.model_predictive_rl.do_action_clip = True
    if planning_width is not None:
        policy_config.model_predictive_rl.planning_width = planning_width
        policy_config.model_predictive_rl.do_action_clip = True
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
    parser = argparse.ArgumentParser(description='Random / Tree / Diffusion 表格对比（TABLE I + TABLE II）')
    parser.add_argument('-m', '--model_dir', type=str, default='logs/debug')
    parser.add_argument('--output_dir', type=str, default='logs/comparison')
    parser.add_argument('--n_episodes', type=int, default=5)
    parser.add_argument('--random_config', type=str, default='configs/infocom_benchmark/random.py')
    parser.add_argument('--datasets', type=str, nargs='+', default=['Purdue'],
                        help='Purdue, NCSU, KAIST 中选一个或多个')
    parser.add_argument('--compare_diffusion', action='store_true', help='加入 Diffusion 策略')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--seed', type=int, default=0, help='单种子时使用')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='多种子时使用，报 mean±std，如 --seeds 0 1 2 3 4')
    parser.add_argument('--planning_depths', type=int, nargs='*', default=None,
                        help='Ablation: Tree 的 depth 列表，如 --planning_depths 1 2')
    parser.add_argument('--planning_widths', type=int, nargs='*', default=None,
                        help='Ablation: Tree 的 width 列表，如 --planning_widths 5')
    parser.add_argument('--baselines', type=str, nargs='*', default=['stay', 'greedy_aoi', 'nearest_high_aoi'],
                        help='Rule-based baseline：stay, greedy_aoi, nearest_high_aoi；空则不加')
    args = parser.parse_args()
    if args.seeds is None:
        args.seeds = [args.seed]

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda:0' if args.gpu and torch.cuda.is_available() else 'cpu')
    run_args = _make_args(args.output_dir)

    config_our = os.path.join(args.model_dir, 'config.py')
    weights_our = os.path.join(args.model_dir, 'best_val.pth')
    weights_diff = os.path.join(args.model_dir, 'best_val_with_diffusion.pth')

    # Tree 变体：若指定 planning_depths/widths 则生成 Tree_d1_w5 等
    from itertools import product
    tree_variants = []
    if args.planning_depths or args.planning_widths:
        depths = args.planning_depths if args.planning_depths else [None]
        widths = args.planning_widths if args.planning_widths else [None]
        for d, w in product(depths, widths):
            key = 'Tree'
            if d is not None:
                key += f'_d{d}'
            if w is not None:
                key += f'_w{w}'
            tree_variants.append((key, d, w))
    else:
        tree_variants = [('Tree', None, None)]

    baseline_list = []  # (config_path, method_key)
    for bl in args.baselines:
        if bl == 'stay':
            baseline_list.append(('configs/infocom_benchmark/stay.py', 'Stay'))
        elif bl == 'greedy_aoi':
            baseline_list.append(('configs/infocom_benchmark/greedy_aoi.py', 'GreedyAoI'))
        elif bl == 'nearest_high_aoi':
            baseline_list.append(('configs/infocom_benchmark/nearest_high_aoi.py', 'NearestHighAoI'))

    methods = ['Random'] + [k for k, _, _ in tree_variants]
    if args.compare_diffusion:
        methods.append('Diffusion')
    methods += [k for _, k in baseline_list]

    # 多种子时按 seed 收集，再聚合 mean±std
    table1_by_seed = {s: {d: {} for d in args.datasets} for s in args.seeds}
    table2_by_seed = {s: {d: {} for d in args.datasets} for s in args.seeds}

    for seed in args.seeds:
        set_random_seeds(seed)
        logging.info('========== Seed %s ==========', seed)
        for dataset in args.datasets:
            logging.info('========== Dataset: %s ==========', dataset)
            set_env_dataset(dataset)

            # Tree (含 ablation 变体)
            for method_key, pdepth, pwidth in tree_variants:
                if os.path.isfile(config_our) and os.path.isfile(weights_our):
                    env_our, agent_our, explorer_our = _setup_env_and_policy(
                        device, config_our, model_weights=weights_our, dataset=dataset,
                        planning_depth=pdepth, planning_width=pwidth)
                    list_aoi, list_cov, list_energy, list_data, list_time = [], [], [], [], []
                    for _ in range(args.n_episodes):
                        explorer_our.run_k_episodes(k=1, phase='test', args=run_args, plot_index=0)
                        if explorer_our.statistics is not None:
                            _, _, aoi, energy, data, cov = explorer_our.statistics
                            list_aoi.append(float(aoi))
                            list_cov.append(float(cov))
                            list_energy.append(float(energy))
                            list_data.append(float(data))
                        if getattr(explorer_our, 'step_time_ms', None) is not None:
                            list_time.append(explorer_our.step_time_ms)
                    if list_aoi:
                        table1_by_seed[seed][dataset][method_key] = (
                            sum(list_aoi) / len(list_aoi),
                            sum(list_cov) / len(list_cov),
                            sum(list_energy) / len(list_energy),
                            sum(list_data) / len(list_data),
                        )
                    if list_time:
                        table2_by_seed[seed][dataset][method_key] = sum(list_time) / len(list_time)

            # Diffusion
            if 'Diffusion' in methods and os.path.isfile(config_our):
                env_diff, agent_diff, explorer_diff = _setup_env_and_policy(
                    device, config_our, model_weights=weights_our, dataset=dataset)
                policy_diff = getattr(explorer_diff.robot, 'policy', None)
                if policy_diff is not None and hasattr(policy_diff, 'enable_diffusion'):
                    policy_diff.enable_diffusion()
                    load_file = weights_diff if os.path.isfile(weights_diff) else weights_our
                    if os.path.isfile(load_file):
                        try:
                            ckpt = torch.load(load_file, map_location=device)
                            if isinstance(ckpt, dict) and 'diffusion_model' in ckpt:
                                policy_diff.diffusion_model.load_state_dict(ckpt['diffusion_model'])
                        except Exception as e:
                            logging.warning('Could not load diffusion: %s', e)
                list_aoi, list_cov, list_energy, list_data, list_time = [], [], [], [], []
                for _ in range(args.n_episodes):
                    explorer_diff.run_k_episodes(k=1, phase='test', args=run_args, plot_index=0)
                    if explorer_diff.statistics is not None:
                        _, _, aoi, energy, data, cov = explorer_diff.statistics
                        list_aoi.append(float(aoi))
                        list_cov.append(float(cov))
                        list_energy.append(float(energy))
                        list_data.append(float(data))
                    if getattr(explorer_diff, 'step_time_ms', None) is not None:
                        list_time.append(explorer_diff.step_time_ms)
                if list_aoi:
                    table1_by_seed[seed][dataset]['Diffusion'] = (
                        sum(list_aoi) / len(list_aoi),
                        sum(list_cov) / len(list_cov),
                        sum(list_energy) / len(list_energy),
                        sum(list_data) / len(list_data),
                    )
                if list_time:
                    table2_by_seed[seed][dataset]['Diffusion'] = sum(list_time) / len(list_time)

            # Random
            if os.path.isfile(args.random_config):
                env_rand, agent_rand, explorer_rand = _setup_env_and_policy(
                    device, args.random_config, dataset=dataset)
                list_aoi, list_cov, list_energy, list_data, list_time = [], [], [], [], []
                for _ in range(args.n_episodes):
                    explorer_rand.run_k_episodes(k=1, phase='test', args=run_args, plot_index=0)
                    if explorer_rand.statistics is not None:
                        _, _, aoi, energy, data, cov = explorer_rand.statistics
                        list_aoi.append(float(aoi))
                        list_cov.append(float(cov))
                        list_energy.append(float(energy))
                        list_data.append(float(data))
                    if getattr(explorer_rand, 'step_time_ms', None) is not None:
                        list_time.append(explorer_rand.step_time_ms)
                if list_aoi:
                    table1_by_seed[seed][dataset]['Random'] = (
                        sum(list_aoi) / len(list_aoi),
                        sum(list_cov) / len(list_cov),
                        sum(list_energy) / len(list_energy),
                        sum(list_data) / len(list_data),
                    )
                if list_time:
                    table2_by_seed[seed][dataset]['Random'] = sum(list_time) / len(list_time)

            # Baselines: Stay, Greedy (AoI), Nearest High AoI（每次用唯一 module_name 避免 config 被覆盖导致同策略）
            for config_path, method_key in baseline_list:
                if not os.path.isfile(config_path):
                    continue
                env_bl, agent_bl, explorer_bl = _setup_env_and_policy(
                    device, config_path, dataset=dataset,
                    module_name='config_baseline_%s' % method_key)
                list_aoi, list_cov, list_energy, list_data, list_time = [], [], [], [], []
                for _ in range(args.n_episodes):
                    explorer_bl.run_k_episodes(k=1, phase='test', args=run_args, plot_index=0)
                    if explorer_bl.statistics is not None:
                        _, _, aoi, energy, data, cov = explorer_bl.statistics
                        list_aoi.append(float(aoi))
                        list_cov.append(float(cov))
                        list_energy.append(float(energy))
                        list_data.append(float(data))
                    if getattr(explorer_bl, 'step_time_ms', None) is not None:
                        list_time.append(explorer_bl.step_time_ms)
                if list_aoi:
                    table1_by_seed[seed][dataset][method_key] = (
                        sum(list_aoi) / len(list_aoi),
                        sum(list_cov) / len(list_cov),
                        sum(list_energy) / len(list_energy),
                        sum(list_data) / len(list_data),
                    )
                if list_time:
                    table2_by_seed[seed][dataset][method_key] = sum(list_time) / len(list_time)

    # 聚合多种子 -> mean (及 std)
    method_order = ['Random'] + [k for k, _, _ in tree_variants] + (['Diffusion'] if args.compare_diffusion else []) + [k for _, k in baseline_list]
    method_order = [m for m in method_order if m in methods]
    table1 = {d: {} for d in args.datasets}
    table2 = {d: {} for d in args.datasets}
    table1_std = {d: {} for d in args.datasets}
    table2_std = {d: {} for d in args.datasets}
    for dataset in args.datasets:
        for method in method_order:
            t1_vals = [table1_by_seed[s][dataset].get(method) for s in args.seeds if table1_by_seed[s][dataset].get(method) is not None]
            t2_vals = [table2_by_seed[s][dataset].get(method) for s in args.seeds if table2_by_seed[s][dataset].get(method) is not None]
            if t1_vals:
                arr = np.array(t1_vals)
                table1[dataset][method] = tuple(np.mean(arr, axis=0))
                table1_std[dataset][method] = tuple(np.std(arr, axis=0)) if len(t1_vals) > 1 else (0, 0, 0, 0)
            if t2_vals:
                table2[dataset][method] = np.mean(t2_vals)
                table2_std[dataset][method] = np.std(t2_vals) if len(t2_vals) > 1 else 0
    report_std = len(args.seeds) > 1

    # 输出 LaTeX 与 CSV（表内用显示名；多种子时 mean±std）
    def fmt(x, decimals=3):
        if isinstance(x, float):
            return f'{x:.{decimals}f}'
        return str(x)

    out_dir = args.output_dir
    base = os.path.join(out_dir, 'table_comparison')

    # TABLE I LaTeX
    latex1 = []
    latex1.append('\\begin{table}[t]')
    latex1.append('\\caption{Impact of different policies. (R: Mean AoI, $\\psi$: Coverage, $\\rho$: Energy, $\\varsigma$: Data amount)' +
                  (' Mean $\\pm$ std over seeds.' if report_std else '') + '}')
    latex1.append('\\label{tab:policy_impact}')
    latex1.append('\\centering')
    col_spec = 'l' + 'r' * 4 * len(args.datasets)
    latex1.append('\\begin{tabular}{' + col_spec + '}')
    header = 'Method & ' + ' & '.join(
        [f'\\multicolumn{{4}}{{c}}{{{d}}}' for d in args.datasets]) + ' \\\\'
    latex1.append(header)
    latex1.append(' & ' + ' & '.join(
        ['R & $\\psi$ & $\\rho$ & $\\varsigma$'] * len(args.datasets)) + ' \\\\ \\hline')
    for method in method_order:
        row = _display_name(method)
        for d in args.datasets:
            if method in table1.get(d, {}):
                r, psi, rho, var = table1[d][method]
                if report_std and method in table1_std.get(d, {}):
                    sr, sp, srho, svar = table1_std[d][method]
                    row += f' & {fmt(r)} $\\pm$ {fmt(sr)} & {fmt(psi)} $\\pm$ {fmt(sp)} & {fmt(rho)} $\\pm$ {fmt(srho)} & {fmt(var)} $\\pm$ {fmt(svar)}'
                else:
                    row += f' & {fmt(r)} & {fmt(psi)} & {fmt(rho)} & {fmt(var)}'
            else:
                row += ' & - & - & - & -'
        row += ' \\\\'
        latex1.append(row)
    latex1.append('\\end{tabular}')
    latex1.append('\\end{table}')
    path_latex1 = base + '_table1.tex'
    with open(path_latex1, 'w') as f:
        f.write('\n'.join(latex1))
    logging.info('TABLE I LaTeX saved: %s', path_latex1)

    # TABLE II LaTeX
    latex2 = []
    latex2.append('\\begin{table}[t]')
    latex2.append('\\caption{Computational complexity by time cost (ms, per step).' +
                  (' Mean $\\pm$ std over seeds.' if report_std else '') + '}')
    latex2.append('\\label{tab:time_cost}')
    latex2.append('\\centering')
    latex2.append('\\begin{tabular}{l' + 'r' * len(args.datasets) + '}')
    latex2.append('Method & ' + ' & '.join(args.datasets) + ' \\\\ \\hline')
    for method in method_order:
        row = _display_name(method)
        for d in args.datasets:
            t = table2.get(d, {}).get(method)
            t_std = table2_std.get(d, {}).get(method) if report_std else None
            if t is not None:
                if t_std is not None and t_std > 0:
                    row += f' & {fmt(t, 2)} $\\pm$ {fmt(t_std, 2)}'
                else:
                    row += f' & {fmt(t, 2)}'
            else:
                row += ' & -'
        row += ' \\\\'
        latex2.append(row)
    latex2.append('\\end{tabular}')
    latex2.append('\\end{table}')
    path_latex2 = base + '_table2.tex'
    with open(path_latex2, 'w') as f:
        f.write('\n'.join(latex2))
    logging.info('TABLE II LaTeX saved: %s', path_latex2)

    # CSV（显示名 + 可选 std 列）
    import csv
    with open(base + '_table1.csv', 'w', newline='') as f:
        w = csv.writer(f)
        cols = ['Method'] + [f'{d}_{m}' for d in args.datasets for m in ['R', 'psi', 'rho', 'varsigma']]
        if report_std:
            cols += [f'{d}_{m}_std' for d in args.datasets for m in ['R', 'psi', 'rho', 'varsigma']]
        w.writerow(cols)
        for method in method_order:
            row = [_display_name(method)]
            for d in args.datasets:
                if method in table1.get(d, {}):
                    row.extend(table1[d][method])
                else:
                    row.extend(['', '', '', ''])
            if report_std:
                for d in args.datasets:
                    if method in table1_std.get(d, {}):
                        row.extend(table1_std[d][method])
                    else:
                        row.extend(['', '', '', ''])
            w.writerow(row)
    with open(base + '_table2.csv', 'w', newline='') as f:
        w = csv.writer(f)
        cols2 = ['Method'] + list(args.datasets)
        if report_std:
            cols2 += [f'{d}_std' for d in args.datasets]
        w.writerow(cols2)
        for method in method_order:
            row = [_display_name(method)] + [table2.get(d, {}).get(method, '') for d in args.datasets]
            if report_std:
                row += [table2_std.get(d, {}).get(method, '') for d in args.datasets]
            w.writerow(row)
    logging.info('CSV saved: %s_table1.csv, %s_table2.csv', base, base)

    # 画两张表为 PNG（matplotlib table）
    try:
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']

        # Table I figure: Method | for each dataset: R, ψ, ρ, ς (4 cols per dataset)
        ncol = 1 + 4 * len(args.datasets)
        fig1, ax1 = plt.subplots(figsize=(min(14, 2 * ncol), 2.5))
        ax1.set_axis_off()
        header1 = ['Method'] + [x for d in args.datasets for x in ['R', 'ψ', 'ρ', 'ς']]
        c1 = [header1]
        for method in method_order:
            row = [_display_name(method)]
            for d in args.datasets:
                if method in table1.get(d, {}):
                    r1, p, rho, v = table1[d][method]
                    if report_std and method in table1_std.get(d, {}):
                        sr, sp, srho, sv = table1_std[d][method]
                        row.extend([f'{r1:.3f}±{sr:.3f}', f'{p:.3f}±{sp:.3f}', f'{rho:.3f}±{srho:.3f}', f'{v:.3f}±{sv:.3f}'])
                    else:
                        row.extend([f'{r1:.3f}', f'{p:.3f}', f'{rho:.3f}', f'{v:.3f}'])
                else:
                    row.extend(['-', '-', '-', '-'])
            c1.append(row)
        tab1 = ax1.table(cellText=c1[1:], colLabels=c1[0], loc='center', cellLoc='center')
        tab1.auto_set_font_size(False)
        tab1.set_fontsize(9)
        tab1.scale(1.2, 2)
        plt.title('TABLE I: Impact of different policies (R, ψ, ρ, ς)')
        plt.tight_layout()
        plt.savefig(base + '_table1.png', dpi=150, bbox_inches='tight')
        plt.close()

        # Table II figure
        fig2, ax2 = plt.subplots(figsize=(2 + 2 * len(args.datasets), 2))
        ax2.set_axis_off()
        c2 = [['Method'] + list(args.datasets)]
        for method in method_order:
            r = [_display_name(method)]
            for d in args.datasets:
                t = table2.get(d, {}).get(method)
                t_std = table2_std.get(d, {}).get(method) if report_std else None
                if t is not None:
                    if t_std is not None and t_std > 0:
                        r.append(f'{t:.2f}±{t_std:.2f}')
                    else:
                        r.append(f'{t:.2f}')
                else:
                    r.append('-')
            c2.append(r)
        tab2 = ax2.table(cellText=c2[1:], colLabels=c2[0], loc='center', cellLoc='center')
        tab2.auto_set_font_size(False)
        tab2.set_fontsize(10)
        tab2.scale(1.2, 2)
        plt.title('TABLE II: Time cost (ms) per step')
        plt.tight_layout()
        plt.savefig(base + '_table2.png', dpi=150, bbox_inches='tight')
        plt.close()
        logging.info('Table figures saved: %s_table1.png, %s_table2.png', base, base)
    except Exception as e:
        logging.warning('Could not save table figures: %s', e)

    logging.info('Done. TABLE I = metrics (R, ψ, ρ, ς), TABLE II = time (ms).')


if __name__ == '__main__':
    main()
