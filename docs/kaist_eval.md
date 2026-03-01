# 在 KAIST 数据集上做对比与可视化

KAIST 数据集配置已在 `configs/config.py` 中（92 users，sensing_range=220）。数据文件路径：`envs/crowd_sim/dataset/KAIST/92 users.csv`。

## 1. 单策略测试 + 全部可视化（KAIST）

在 KAIST 上跑 Our Policy 并生成所有测试阶段可视化图（轨迹、AoI 分布、能量曲线等）：

```bash
python test_our_policy.py -m logs/debug --dataset KAIST --output_dir logs/kaist_test
```

可选：生成 HTML 轨迹页并保存到同一目录：

```bash
python test_our_policy.py -m logs/debug --dataset KAIST --output_dir logs/kaist_test --vis_html --plot_loop --moving_line
```

生成的文件在 `logs/kaist_test/` 下，例如：
- `our_policy_episode_metrics.png`
- `our_policy_timeseries.png`
- `our_policy_users_and_uav_trajectories.png`
- `our_policy_uav_energy.png`、`our_policy_aoi_distribution.png`、`our_policy_spatial_aoi.png`、`our_policy_trajectory_colored_by_time.png`、`our_policy_snapshots.png`
- 若加 `--vis_html`：`test_page_*.html`

## 2. 策略对比图（KAIST）

在 KAIST 上跑 Our Policy / Random / Stay / Greedy / Nearest High AoI（及可选的 Diffusion），并生成对比图：

```bash
python run_comparison.py -m logs/debug --output_dir logs/comparison_KAIST --dataset KAIST --n_episodes 5
```

可选：加入 Diffusion 与 Greedy-Diffusion，以及多种子：

```bash
python run_comparison.py -m logs/debug --output_dir logs/comparison_KAIST --dataset KAIST --n_episodes 5 --compare_diffusion --compare_greedy_diffusion
```

图会保存在 `logs/comparison_KAIST/`：
- `policy_comparison.png`、`policy_comparison_normalized.png`
- `comparison_aoi.png`、`comparison_coverage.png`、`comparison_energy.png`、`comparison_data_amount.png`
- `comparison_radar.png`、`comparison_with_std.png`

## 3. 表格对比 TABLE I / TABLE II（仅 KAIST）

在 KAIST 上生成论文风格的两张表（R, ψ, ρ, ς 与每步耗时），并保存 CSV 与 PNG：

```bash
python run_table_comparison.py -m logs/debug --output_dir logs/comparison_KAIST --datasets KAIST --n_episodes 5
```

可选：加入 Diffusion、多种子、baseline：

```bash
python run_table_comparison.py -m logs/debug --output_dir logs/comparison_KAIST --datasets KAIST --n_episodes 5 --compare_diffusion --seeds 0 1 2 --baselines stay greedy_aoi nearest_high_aoi
```

输出在 `logs/comparison_KAIST/`：
- `table_comparison_table1.csv` / `table_comparison_table2.csv`
- `table_comparison_table1.png` / `table_comparison_table2.png`

## 4. 训练曲线（与数据集无关）

若你在某次训练时已指定 `--output_dir`（例如 `logs/debug`），训练曲线来自 TensorBoard，与当前评估数据集无关。导出 PNG：

```bash
python plot_training_curves.py --logdir logs/debug --output_dir logs/debug/curves
```

## 前提

- 已有一份训练好的模型（如 `logs/debug` 下有 `config.py`、`best_val.pth`）。训练时可用任意数据集（如 Purdue）；评估时用 `--dataset KAIST` 即可在 KAIST 上评估。
- KAIST 数据文件存在：`envs/crowd_sim/dataset/KAIST/92 users.csv`。
