# Electric Vehicle Routing with Clarke & Wright + Proximal Policy Optimization

> A hybrid research / engineering project exploring how a classical constructive heuristic and a modern deep reinforcement learning policy can cooperate (and compete) to solve the Electric Vehicle Routing Problem (EVRP) with battery, charging, capacity, and multi‑vehicle constraints.

---
## 🚀 Why This Project Exists
Traditional VRP heuristics (e.g., Clarke & Wright) are lightning‑fast but rigid. Deep RL (e.g., PPO) is adaptive but data‑ and tuning‑hungry. EV fleets add *energy feasibility*, *charging strategy*, and *state‑of‑charge safety* to the already NP‑hard VRP. This repo blends both worlds:

| Track | Strength | Limitation | Role Here |
|-------|----------|------------|-----------|
| Clarke & Wright (C&W) | Deterministic, fast baseline, interpretable merges | No learned adaptation; sensitive to feasibility filters | Provides a reference lower/upper performance band + feasible seed routes |
| PPO Policy | Learns implicit trade‑offs; generalizable | Needs reward shaping + stable training | Explores new visitation & charging strategies; can surpass heuristic in tailored regimes |

---
## 🧩 Problem Setting
We solve EVRP on a Euclidean plane:
- Depot node `0`
- Customers with positive demand (weight / load)
- Optional charging stations
- Vehicles have: load capacity `Q`, battery capacity `B (Wh)`, minimum SOC fraction `m` (reserve), consumption rate `ρ (Wh/km)`.
- Objective (implicit): Serve all positive‑demand customers feasibly while minimizing a composite cost (distance/time/penalties) + maximizing coverage rewards.

A solution is a *set of vehicle tours*: each begins/ends at depot, may insert charging nodes, respects load and energy feasibility constraints.

---
## 🧠 Core Algorithms
### 1. Clarke & Wright Savings (EVRP Extension)
Standard savings \( S(i,j) = d(0,i) + d(0,j) - d(i,j) \) sorted descending. Merges allowed only if:
- Combined load ≤ `Q`
- Battery-feasible after (possible) insertion of a charging station between critical legs
- Max vehicle constraint not violated (or post‑pruned)
- Unserved customers later patched by single-customer feasible routes

**Battery Feasibility Check** (high‑level):
1. Simulate energy decrement for each leg.
2. If SOC would fall below `m * B`, attempt insertion of a charging station `c` minimizing detour and restoring SOC.
3. Reject merge if no feasible insertion.

### 2. Proximal Policy Optimization (PPO)
Policy learns a node selection strategy under a shaped reward signal.

**State Vector** (length = `N + 1 + N`):
```
[ one_hot(current_node) | battery_scalar | visited_mask ]
```
**Action Space**: discrete node index (customers, depot, charging stations). Invalid / undesirable actions masked.

**Network**: 3 residual‐enhanced linear blocks + LayerNorm + Dropout → policy head (softmax) & value head.

**Objective**:
```
L = L_clip + 0.5 * value_loss - beta * entropy
```
with gradient clipping, advantage normalization, and LR scheduling on validation reward.

**Reward Shaping Components**:
| Component | Purpose | Sign |
|-----------|---------|------|
| Distance penalty | Shrink route length | − |
| First-visit bonus | Coverage acceleration | + |
| Completion bonus | Encourage full solution | + |
| Detour/revisit penalty | Avoid loops | − |
| Illegal/infeasible penalty | Enforce constraints | − |
| Step penalty | Stop dithering | − |
| (Planned) Charging incentives | Strategic energy mgmt | ± |
| (Planned) Dispatch / charge costs | Multi-vehicle balance | ± |

---
## 🗂️ Repository Structure
```
.
├── train.py                     # PPO training loop (episodes, logging, checkpointing)
├── evaluate.py                  # Unified evaluation (choose heuristic or PPO interactive run)
├── evaluate_ppo.py              # Adaptive PPO model loading & baseline comparison
├── mp_utils.py                  # PPOPolicy, Clarke & Wright algorithm, EV environment
├── make_gif.py                  # Visualization / GIF generation of tours
├── test_case_evrp.tsp           # Example EVRP instance (coordinates + descriptors)
├── best_ppo_policy.pth          # Best validation snapshot
├── ppo_checkpoint.pth(.meta.json) # Resume state + metadata
├── ppo_training_log.csv         # Episode-wise training metrics
├── ppo_training_curves.png      # Reward/Loss plot
├── ppo_training_distance_customers.png # Distance & coverage plot
├── data/
│   ├── cw_solution_stats.txt    # Heuristic stats
│   ├── ppo_solution_stats.txt   # PPO stats
│   ├── cw_solution.gif          # Route animation (heuristic)
│   └── ppo_solution.gif         # Route animation (PPO)
└── README.md / README_CLEAN.md
```

---
## ⚙️ Quick Start
### 1. Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch numpy matplotlib pandas
```

### 2. Train PPO
```powershell
python train.py --num_nodes 5 --num_samples 100 --num_episodes 200
```
Resume later:
```powershell
python train.py --resume
```

### 3. Evaluate
Interactive (choose mode):
```powershell
python evaluate.py
```
Standalone adaptive PPO evaluator:
```powershell
python evaluate_ppo.py --model best_ppo_policy.pth
```

### 4. Visuals
GIFs placed in `data/`; metrics plots auto‑generated after training.

---
## 🔍 Detailed Methodology
### State & Action Mask
- Depot masked until coverage complete.
- Customers masked once served.
- Charging stations should remain available (plan: remove them from permanent “visited” banning for recharge loops).

### PPO Training Loop (train.py)
1. Batch generate random instances.
2. For each training instance: rollout until terminal or step cap.
3. Store `(state, action, log_prob, value, reward)` sequences.
4. After full episode: compute returns & advantages.
5. Perform PPO epochs with minibatching.
6. Log metrics; early stopping + LR reduction using validation slice.

### Adaptive Loading (evaluate_ppo.py)
Infers original dimension from saved layer shapes. If test instance larger, optionally truncates coordinates/demands to match. Loads existing weights with `strict=False`, preserving learned embedding of prior state layout.

### Clarke & Wright Enhancement Pipeline
1. Init atomic routes.
2. Compute & sort savings.
3. Attempt merges; run battery feasibility insertion logic.
4. Vehicle cap reconciliation.
5. Single-customer recovery of any unserved positive demand.
6. Optional final merges for densification.

### Diagnostics (evaluate.py)
Post-run heuristic analysis enumerates unserved customers, classifies root cause (capacity / energy / charger proximity / vehicle limit), suggests remedy (increase battery, add charger, raise max vehicles, etc.).

---
## 📊 Logging & Metrics
`ppo_training_log.csv` columns:
| Column | Meaning |
|--------|---------|
| episode | 1-based index |
| train_reward / val_reward | Mean shaped rewards |
| loss | Combined (policy + value) average |
| lr | Current learning rate |
| train_distance | Mean cumulative path distance (approx) |
| train_customers | Mean customers visited (coverage) |
| policy_loss / value_loss | PPO sub-losses |
| entropy | Average sampling entropy (policy stochasticity) |
| steps | Total collected steps for update |

Plots auto‑generated at completion (or upon manual close) for reward/loss + distance/coverage.

---
## ⚖️ Reward Shaping Guidance
If coverage is slow: *increase* first-visit bonus or completion bonus.
If detours/loops: *increase* detour penalty or step penalty.
If agent quits early (premature depot): raise completion bonus relative to cumulative distance penalty of finishing all customers.
If no charging: add conditional bonus for charging at low SOC and *never* mask charging stations.

---
## 🔋 Multi‑Vehicle vs Charging Balance (Planned)
Without additional economics, one charged vehicle may dominate. Add:
- Dispatch penalty per new vehicle (discourages explosion, but small enough to still use >1).
- Per-charge penalty + escalating factor.
- Max charges per vehicle (soft → hard cap).
- Route duration soft penalty (time proportional to distance / speed).
These features create a *Pareto trade‑off* learned by the policy.

---
## 🧪 Reproducibility Tips
- Set seeds (not yet wired): `torch.manual_seed`, `np.random.seed`.
- Log config metadata (already in checkpoint `.meta.json`).
- Freeze reward weights when comparing variants.

---
## 🧯 Troubleshooting
| Symptom | Cause | Fix |
|---------|-------|-----|
| Flat reward curves | Wrong CSV header or plotting empty file | Header auto‑repair; ensure restart not appending corrupted rows |
| Customers visited always 0 | Visit counting after state update | Capture pre-step visited flag (already patched) |
| No charging events | Charging stations masked as visited | Exempt stations from mask; add charging incentive |
| Dimension mismatch loading model | Different node count | Adaptive loading (truncate or extend) |
| NaN loss | Advantage explosion / zero std | Clamp std + check reward scaling |

---
## 🛣️ Roadmap
- [ ] Charging incentive & economic balancing layer
- [ ] GAE (λ) advantages for lower variance
- [ ] Attention-based policy (scales to larger N)
- [ ] Heterogeneous fleets (mixed capacities / batteries)
- [ ] On-policy curriculum (dynamic expansion of node set)
- [ ] Hybrid local search refinement (2‑opt after PPO rollout)

---
## 🧾 Citation (Example)
If you build on this work:
```
@misc{evrp-cw-ppo-2025,
  title  = {Hybrid Clarke & Wright + PPO for Electric Vehicle Routing},
  author = {Your Name},
  year   = {2025},
  note   = {Git repository: EVRP with C&W and PPO}
}
```

---
## 🙏 Acknowledgements
Inspired by foundational VRP heuristics, the PPO algorithm (Schulman et al.), and recent EVRP surveys. Thanks to open-source PyTorch and the scientific Python ecosystem.

---
## 📎 Appendix A: Clarke & Wright (EVRP) Pseudocode
```
R = { [0,i,0] for each customer i }
Compute S(i,j) for all i<j
Sort savings descending
for (i,j) in savings:
  if route_i tail is i and route_j head is j and loads feasible:
     candidate = merge(route_i, route_j)
     if battery_feasible_with_charging_insert(candidate):
         commit merge
Post-process: enforce max vehicles (merge / prune)
Add single-customer routes for unserved
Return routes
```

### Battery Feasibility with Charging Insert (Sketch)
```
E = B
for (u->v) in route:
  need = ρ * d(u,v)
  if E - need >= mB:
     E -= need
  else:
     for c in charging_stations:
        if feasible(u->c->v): insert c, E = B - ρ*d(c,v); break
     if not inserted: return False
return True
```

---
## 📎 Appendix B: PPO Update Pseudocode
```
Collect trajectories: (s_t, a_t, r_t, logp_old_t, V_t)
Compute returns G_t (reverse discounted sum)
A_t = G_t - V_t; normalize A
for epoch in 1..E:
  shuffle indices
  for minibatch M:
     ratio = exp(logπ_new - logπ_old)
     surr1 = ratio * A
     surr2 = clip(ratio, 1-ε, 1+ε) * A
     policy_loss = -mean(min(surr1,surr2)) - β * entropy
     value_loss  = MSE(G, V_new)
     total = policy_loss + 0.5*value_loss
     backprop + grad clip
```

---
## 📎 Appendix C: Adaptive Loading Logic
1. Load state dict (strict=False)
2. Derive original action_dim from `policy_head.weight.shape`
3. If current instance larger: optionally truncate positions/demands
4. Rebuild model with inferred (state_dim, action_dim) and copy overlapping weights

---
## 📎 Appendix D: Reward Component Tuning Table (Example)
| Component | Default | Range to Experiment |
|-----------|---------|---------------------|
| distance_weight | -1.0 | [-0.2, -2.0] |
| visit_customer_reward | 0.2 | [0.1, 1.0] |
| completion_bonus | 2.0 | [2, 10] |
| illegal_penalty | -1.0 | [-5, -0.5] |
| detour_penalty | -0.2 | [-1.0, -0.1] |
| step_penalty | -0.01 | [-0.05, -0.005] |
| entropy_coef | 0.02 | [0.005, 0.05] |

---
## ✅ Final Notes
This README balances *narrative clarity* with *technical fidelity*. Adjust reward weights and masking rules early—those dominate learning stability. Consider enabling charging station revisits next and adding per-charge economics for richer behavior.

Happy routing & reinforcement learning! 🛣️🔋🤖
