# 如何提升 Diffusion 策略性能

## 已实现、可直接用的方法

### 1. 离散化到树动作（默认已开）
- **配置**：`configs/infocom_benchmark/mp_separate_dp.py` 里 `diffusion_discretize_output = True`
- **作用**：Diffusion 采样的连续动作会 snap 到与 Tree 相同的离散动作，再按 return 选最优，避免连续偏差。
- **建议**：保持开启。

### 2. 多收集 BC 数据 + 多训几轮
```bash
python train_diffusion_bc.py -m logs/debug --output_dir logs/debug --steps 15000 --epochs 200
```
- 提高 `--steps`（如 15000～20000）和 `--epochs`（如 200～300）可降低模仿误差。

### 3. 推理时多采样
- 在 `configs/infocom_benchmark/mp_separate_dp.py` 里把 `diffusion_num_samples` 改为 **64 或 128**（更稳、更慢）。

### 4. DAgger（减轻状态分布偏移）
- 第 1 轮：用 Tree 收集 (s, a*) 训 Diffusion。
- 第 2 轮起：用「当前 Diffusion + Tree」混合控制 env（部分步用 Diffusion 走，访问 Diffusion 会到的状态），每步仍用 Tree 标 label，再继续训 Diffusion。
```bash
python train_diffusion_bc.py -m logs/debug --output_dir logs/debug --dagger_rounds 3 --steps 5000 --epochs 120 --dagger_epsilon 0.5
```
- `--dagger_rounds 3`：共 3 轮，后 2 轮用 diffusion 混合 rollout。
- `--dagger_epsilon 0.5`：混合 rollout 时 50% 步用 Diffusion、50% 用 Tree。

**DAgger 耗时说明：** 每轮 = 「收集 steps 条 (s,a)」+ 「训 epochs 轮」。收集时若用 Diffusion 步进（第 2 轮起约一半），每步要采样 32 条轨迹并 rollout，比 Tree 慢很多；训练则每轮在**累积**数据上训满 epochs，数据量逐轮增大。默认 `--steps 8000 --epochs 150` 且 3 轮时，在 CPU 上可能需 **5～15 小时**。若想先快速看效果，建议用「快速 DAgger」：
```bash
# 快速 DAgger：2 轮、每轮 3000 步、80 epoch，约 1～2 小时（视机器而定）
python train_diffusion_bc.py -m logs/debug --output_dir logs/debug --dagger_rounds 2 --steps 3000 --epochs 80 --dagger_epsilon 0.5
```

---

## 建议尝试顺序

1. 确认 **离散化已开**，跑一次对比看 Diffusion 与 Tree 的差距。
2. **加数据 + 加 epoch**：`--steps 15000 --epochs 200`，再对比。
3. 仍差则跑 **DAgger**：`--dagger_rounds 3 --steps 5000 --dagger_epsilon 0.5`。
4. 最后可把 **diffusion_num_samples** 调到 64 再测一次。

更多原因分析见 `docs/diffusion_vs_tree_analysis.md`。
