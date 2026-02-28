# Rule-based baselines

We compare our policy (GCRL-min(AoI) and Diffusion-based (AoI)) against the following **rule-based baselines**. All of them use the same discrete joint action space as the tree-search policy (Cartesian product of per-UAV actions: stay, move in 8 directions, etc.) and require **no training**.

---

## Stay

**Description:** UAVs do not move. At every step, the policy selects the joint action that corresponds to “no motion” for all UAVs (i.e. each UAV’s displacement is (0, 0)).

**Purpose:** Serves as a **lower bound**: it only covers users that fall within sensing range of the initial UAV positions and does not collect data from users that move out of range. Useful to show that active mobility is necessary for good AoI and coverage.

---

## Greedy (AoI)

**Description:** At each step, the policy evaluates every joint action in the discrete action space. For each candidate action, it computes the **next UAV positions** and then sums the **AoI of all users that lie within sensing range** of any UAV after that move. The policy selects the joint action that **maximizes this sum** (i.e. greedily maximizes the total AoI of covered users in the next step).

**Purpose:** A simple **myopic coverage heuristic**: move so that the set of users covered in the next step has the highest total AoI. No lookahead or value function; useful as a strong rule-based baseline that explicitly targets high-AoI users.

---

## Nearest High AoI

**Description:** Each UAV acts **independently**. For each UAV, the policy (1) finds the user with the **current highest AoI**, and (2) selects the **discrete per-UAV action** (from the same 9 options: stay, 4 cardinal directions, 4 diagonals) that moves that UAV **closest to that user’s position** (minimum distance to the user after the move). The joint action is the combination of these per-UAV choices, rounded to the nearest valid joint action if needed.

**Purpose:** A **decentralized, target-chasing** baseline: each UAV is attracted to the single highest-AoI user. Compared to Greedy (AoI), it does not coordinate across UAVs and does not consider how many users are covered, but is simple and often improves over Stay and Random.

---

## Summary

| Baseline           | Decision rule                                      | Coordination   |
|--------------------|-----------------------------------------------------|----------------|
| **Stay**           | No motion for all UAVs                             | —              |
| **Greedy (AoI)**   | Maximize sum of AoI over users covered next step    | Joint (global) |
| **Nearest High AoI** | Each UAV moves toward the current max-AoI user   | Per-UAV        |

All three are implemented in `policies/` and can be run via `run_comparison.py` or `run_table_comparison.py` (enabled by default with `--baselines`).
