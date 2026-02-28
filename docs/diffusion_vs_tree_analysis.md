# 为什么 Diffusion 比不过 Tree？分析与改进思路

## 1. 当前设定回顾

- **Tree**：每步从 **离散** 动作空间（单 UAV 9 个方向 × 2 个 UAV = 81 种联合动作）中，先用 value 筛出 **top-5**，再在这 5 个里选 return 最大的一个执行。
- **Diffusion**：用 BC 学「树搜索选出的最优动作」的分布，推理时 **采样 32 条** 连续动作轨迹，用 **同一套** state_predictor + value_estimator 算 rollout return，取 return 最大的那条的**第一步**执行。

两者都用同一套 value 和 dynamics，区别主要在：**动作从哪来**（离散 5 个 vs 连续采样 32 个）。

---

## 2. 可能原因分析

### 2.1 动作形式不匹配（离散 vs 连续）

- Tree 的动作**永远**是 `one_uav_action_space` 的笛卡尔积（如 `[30,0]`, `[-21,21]` 等），即 9×9=81 个联合动作里的一个。
- BC 的目标就是这 81（再被 clip 成 5）个**离散点**；Diffusion 学的是「在这些点附近」的连续分布。
- 推理时 Diffusion **连续采样**，得到的是 (dx, dy) 的任意实数。即使最后用 value 挑「最好的一条」，那一条也往往是**接近**某个离散动作，而不是完全等于。  
  - 若 reward/动力学对动作**稍微敏感**（例如碰撞判定、边界），连续的小偏差可能带来略差的 reward。
  - 若 value 估计对「没在训练里见过」的连续动作不够准，ranking 可能略偏。

**结论**：连续输出 + 只选 32 条，有可能在「最好的一条」上仍略逊于 Tree 的精确离散最优。

### 2.2 状态分布偏移（OOD）

- BC 的数据是 **用 Tree 策略** 跑出来的 (s, a*)：状态 s 的分布 = Tree 访问到的状态。
- 测试时若 **用 Diffusion 策略**，前期某一步选错，后面状态分布就会和 Tree 越来越不一样。
- Diffusion 在「Tree 没怎么见过的状态」上，预测容易崩，容易连续多步变差 → 恶性循环。

**结论**：Distribution shift 会让 Diffusion 在真实 rollout 里更容易遇到 OOD 状态，表现不如 Tree。

### 2.3 数据量与模仿误差

- 若 **steps/epochs 不够**，BC 的 MSE 还没压下去，模仿树的最优动作就不够准。
- 树的最优动作是「5 选 1」的 one-hot 式分布；用 MSE 去拟合多模态分布本身就有局限，容易平均成中间值，导致选的动作不够「尖」。

**结论**：数据少或训练不足时，Diffusion 的决策会更模糊，比 Tree 差一截可以预期。

### 2.4 Horizon=1 的局限

- 当前 Tree 与 Diffusion 都是 **horizon=1**（只规划一步）。
- 在这种设定下，Diffusion 的「多步生成」优势用不上，只是「用学到的 P(a|s) 生成候选，再交给 value 选」——若 P(a|s) 学得不如 Tree 的 5 选 1 准，就会输。

---

## 3. 改进思路（可操作）

### 3.1 把 Diffusion 输出离散化到 Tree 动作空间（已实现，可选）

- 思路：Diffusion 仍然采样 32 条连续轨迹；对**第一步**动作，按「每个 UAV 单独」snap 到最近的 `one_uav_action_space` 离散动作，再在这 32 个（去重后）离散动作上算 return、挑最好的执行。
- 效果：  
  - 执行的动作**一定**是 Tree 也会用的离散动作，避免连续小偏差带来的 reward/value 偏差。  
  - 仍用 Diffusion 做「候选生成」，用 value 做「在离散空间里选最优」，相当于用 Diffusion 扩展候选集（32 个离散候选 vs Tree 的 5 个）。
- 使用：在配置里打开 `diffusion_discretize_output = True`（若存在该选项），或见下节代码说明。

### 3.2 多收集 BC 数据 + 多训几轮

- 增加 `train_diffusion_bc.py` 的 `--steps`（如 15000～20000）和 `--epochs`（如 200～300）。
- 目的：降低模仿误差，让 P(a|s) 更接近树的最优动作分布。

### 3.3 推理时多采样

- 在 config 里把 `diffusion_num_samples` 提高到 64 或 128（会变慢）。
- 目的：在「离散化前」或「离散化后」都有更多候选，提高挑到好动作的概率。

### 3.4 DAgger（迭代数据收集，减轻 OOD）

- 第 1 轮：用 Tree 收集 (s, a*) → 训 Diffusion。
- 第 2 轮：用当前 Diffusion（或 ε 混合 Tree）跑环境，遇到的状态 s' 用 **Tree 重新算** a*'，得到 (s', a*') 加入 buffer，再训 Diffusion。
- 重复几轮。这样测试时 Diffusion 遇到的 state 在训练数据里更有代表性。
- 实现要点：在 `train_diffusion_bc.py` 里支持「用当前 diffusion 策略 + 一定概率 tree」做 rollout，同时用 tree 算 label。

### 3.5 拉长 Horizon（进阶）

- 若把 Tree 的 `planning_depth` 改为 2 或 3，`get_action_trajectory` 会返回 2～3 步轨迹；用这些 (s, a_1:H) 训 Diffusion，并设 `diffusion_horizon = 2 或 3`。
- 推理时对每条采样轨迹做 2～3 步 rollout，再按 return 选最优轨迹的首动作。
- 目的：让 Diffusion 学多步配合，可能在某些场景下比单步 Tree 更顺（需要同时调 Tree 的 depth 做公平对比）。

### 3.6 用 Value 做 Guided Sampling（进阶）

- 在 Diffusion 采样过程中用 value 估计做 guidance（例如 classifier guidance），让采样往「高 return」方向偏，而不是纯靠「多采样 + 事后选」。
- 实现成本较高，需要把 value 接到 diffusion 的采样步或做 reward-weighted 的 BC。

---

## 4. 建议的尝试顺序

1. **先开「离散化到 Tree 动作」**：看 Diffusion 是否立刻接近或略超 Tree（若仍略逊，多半是 2.2/2.3）。
2. **加数据、加 epoch、加采样数**：再跑对比，看 gap 是否缩小。
3. 若仍差：考虑 **DAgger** 或 **Horizon=2** 的 BC，并配合离散化。

当前代码里已支持「Diffusion 输出离散化到树动作空间」的可选开关，见配置与 `policies/model_predictive_rl.py` 中的 `diffusion_discretize_output`。
