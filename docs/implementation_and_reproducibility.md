# Implementation Details and Reproducibility

## Implementation Details

### Environment and simulation
- **Episode length:** 120 steps; step duration 15 s (30 min per episode).
- **UAVs:** 2 UAVs; max energy 359,640 J per UAV; 9 discrete actions per UAV (stay, 4 cardinal, 4 diagonal), joint action space size 81.
- **Users:** Trajectories and timestamps from CSV datasets (Purdue: 59 users; NCSU: 33; KAIST: 92). Sensing range: Purdue 23.2 (grid units), NCSU/KAIST 220.
- **Reward:** Linear in mean AoI decrease and energy penalty (see `envs/crowd_sim/crowd_sim.py`).

### GCRL-min(AoI) (tree-search policy)
- **Graph encoder (RGL):** 2 layers, node dim 32, similarity `embedded_gaussian`, skip connection.
- **Value network:** MLP [32, 256, 256, 1]; separate graph for value (no sharing with state predictor).
- **State predictor:** Human motion predictor MLP [32, 256, 256, 4]; separate graph.
- **Planning:** Depth 1, width 5 (top-5 actions by value, then best by 1-step return); action clipping enabled; discount γ = 0.95.

### Diffusion-based (AoI) policy
- **Architecture:** DDPM, horizon 1; state embedding from graph encoder (robot nodes only, flattened); time embedding dim 64; hidden dim 256; 100 diffusion steps; β linear 1e-4 → 0.02.
- **Inference:** Sample 32 trajectories, rollout with same state predictor and value estimator, take first action of trajectory with highest return; optional snap to discrete action space (`diffusion_discretize_output = True`).

### Training (RL for tree-search policy)
- **Episodes:** 200; warmup 20 (no training, fill replay); evaluation every 10 episodes.
- **Replay:** Capacity 50,000; batch size 128; Adam, lr 0.001.
- **Target network:** Update every 30 episodes.
- **Exploration:** ε-greedy; ε from 0.5 to 0.1 over first 150 episodes, then 0.1.

### Training (BC for Diffusion)
- **Data:** Tree-search policy rollout; 8,000 (s, a) pairs by default; optional DAgger (e.g. 3 rounds, ε = 0.5).
- **BC:** 150 epochs; batch 64; Adam, lr 1e-4; MSE on noise prediction.

### Baselines (no training)
- **Stay:** Joint action “no motion” for all UAVs.
- **Greedy (AoI):** At each step, choose joint action that maximizes sum of AoI over users covered at next step.
- **Nearest High AoI:** Each UAV independently moves toward the current highest-AoI user (nearest discrete direction).
- **Random:** Uniform over the 81 joint actions.

---

## Reproducibility

### Software and hardware
- **Python:** 3.8+.
- **Key dependencies:** PyTorch, gym 0.21, pandas, numpy, tensorboard (see `requirements.txt`). No fixed PyTorch version in repo; we used a single GPU or CPU for all experiments.
- **Seeds:** Training and evaluation use configurable seeds (default 0); set in `train_our_policy.py`, `run_comparison.py`, and `run_table_comparison.py`.

### Reproducing main results

1. **Train the tree-search policy (GCRL-min(AoI))**
   ```bash
   python train_our_policy.py --config configs/infocom_benchmark/mp_separate_dp.py --output_dir logs/debug --overwrite
   ```
   This uses the config above (200 episodes, warmup 20, etc.). Checkpoints: `best_val.pth` (best validation), `rl_model.pth` (last).

2. **(Optional) Train Diffusion**
   ```bash
   python train_diffusion_bc.py -m logs/debug --output_dir logs/debug --steps 8000 --epochs 150
   ```
   Produces `best_val_with_diffusion.pth`. For stronger Diffusion: `--steps 15000 --epochs 200` or `--dagger_rounds 3 --steps 5000 --dagger_epsilon 0.5`.

3. **Policy comparison (figures)**
   ```bash
   python run_comparison.py -m logs/debug --output_dir logs/comparison --n_episodes 5
   python run_comparison.py -m logs/debug --output_dir logs/comparison --compare_diffusion --n_episodes 5
   ```
   Saves bar charts (AoI, coverage, energy, data) and optional baseline curves (Stay, Greedy, Nearest High AoI). Default seed 0.

4. **Tables (TABLE I & II)**
   ```bash
   python run_table_comparison.py -m logs/debug --output_dir logs/comparison --compare_diffusion --n_episodes 5
   ```
   For mean ± std over seeds:
   ```bash
   python run_table_comparison.py -m logs/debug --output_dir logs/comparison --compare_diffusion --seeds 0 1 2 3 4 --n_episodes 5
   ```
   Outputs: `table_comparison_table1.tex` / `table2.tex`, `.csv`, and `.png`.

5. **Multiple datasets (Purdue, NCSU, KAIST)**
   ```bash
   python run_table_comparison.py -m logs/debug --output_dir logs/comparison --compare_diffusion --datasets Purdue NCSU KAIST --n_episodes 3
   ```
   Dataset paths and parameters are in `configs/config.py` (and `_apply_dataset_env()` for NCSU/KAIST). Ensure the corresponding CSV files exist under `envs/crowd_sim/dataset/`.

### Config and code locations
- **Env & policy:** `configs/config.py`, `configs/infocom_benchmark/mp_separate_dp.py`.
- **Training:** `train_our_policy.py`, `method/trainer.py`.
- **Diffusion BC / DAgger:** `train_diffusion_bc.py`; Diffusion model: `method/diffusion_model.py`.
- **Evaluation & tables:** `run_comparison.py`, `run_table_comparison.py`; metrics/plots: `method/visualize_metrics.py`.

With the same config files, seeds, and commands above, the reported tables and figures can be reproduced up to hardware and library numerical differences.

---

## Ablation and sensitivity checks

**Idea.** We vary key hyperparameters of the tree-search policy and (optionally) Diffusion, then report the same metrics (R, ψ, ρ, ς and time cost) in the same table format. This shows how sensitive the method is to planning depth/width and supports the chosen default.

**What we vary.**  
- **Planning depth** (tree only): number of lookahead steps when evaluating actions (default 1). Deeper search improves value estimates but increases compute.  
- **Planning width** (tree only): number of top actions kept after the value-based clip (default 5). More width gives a richer candidate set but more rollout cost.

**How we run it.** We use a single trained model (same weights) and only override depth/width at evaluation time. For each combination (e.g. depth ∈ {1, 2}, width ∈ {5}), we run the same number of test episodes per dataset, aggregate metrics, and add one row per variant in TABLE I/II (e.g. “GCRL-min(AoI) (d=1,w=5)”, “GCRL-min(AoI) (d=2,w=5)”). No retraining is required for these ablations.

**Commands.**  
```bash
# Sensitivity to planning depth (e.g. depth 1 vs 2)
python run_table_comparison.py -m logs/debug --output_dir logs/comparison --planning_depths 1 2 --n_episodes 5

# Sensitivity to both depth and width
python run_table_comparison.py -m logs/debug --output_dir logs/comparison --planning_depths 1 2 --planning_widths 5 7 --n_episodes 5
```
The script runs the tree policy once per (depth, width) on each dataset and appends the corresponding rows to the tables. Diffusion and baselines are unchanged; only the tree policy is evaluated under different planning settings.

---

## Reporting variance across seeds

**Idea.** To report stability of results, we run evaluation under multiple random seeds. For each (method, dataset) we compute the **mean** and **standard deviation** of each metric (and of per-step time) across seeds, then report **mean ± std** in the tables and figures.

**What we do.**  
- **Seeds:** We fix a set of seeds (e.g. 0, 1, 2, 3, 4). For each seed we set the RNG (PyTorch and, where used, Python’s `random`) before any rollout.  
- **Per seed:** For each method (Random, GCRL-min(AoI), Diffusion, Stay, Greedy, Nearest High AoI) and each dataset, we run the same number of episodes (e.g. 5), then average the episode-level metrics (R, ψ, ρ, ς) and the mean per-step time to get one number per metric per (method, dataset, seed).  
- **Aggregation:** Across seeds we compute mean and std for each of these numbers. Table cells are then displayed as “mean ± std” (e.g. in LaTeX: `3.40 $\pm$ 0.12`). Figure captions or titles note “Mean ± std over seeds.”

**How we run it.**  
```bash
python run_table_comparison.py -m logs/debug --output_dir logs/comparison --compare_diffusion --seeds 0 1 2 3 4 --n_episodes 5
```
With `--seeds 0 1 2 3 4`, the script runs the full evaluation (all methods, all datasets) once per seed, collects the per-seed averages, then computes mean and std and writes them into TABLE I, TABLE II, the CSVs, and the PNGs. Single-seed runs (default `--seed 0`) report only the mean with no std.
