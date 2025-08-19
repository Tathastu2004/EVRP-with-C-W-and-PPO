# EVRP with Clarke & Wright (C&W) and PPO ✅

Readable, reproducible implementation of the Electric Vehicle Routing Problem (EVRP) solved by:
1. Proximal Policy Optimization (PPO) – learning routing + charging via deep RL.
2. Clarke & Wright Savings heuristic – classic constructive baseline adapted to EV constraints.

---
## 1. Quick Start (Windows / PowerShell)
```powershell
# (Optional) create virtual env if not already present
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch numpy matplotlib pandas

# Train a tiny demo (fast sanity)
python train.py --num_nodes 5 --num_samples 16 --num_episodes 3

# Resume and extend training
python train.py --resume --num_nodes 5 --num_samples 32 --num_episodes 10

# Evaluate a trained PPO model vs C&W
python evaluate_ppo.py --model_path trained_ppo_policy.pth --test_file test_case_evrp.tsp
```

---
## 2. Project Structure
```
mp_utils.py          # PPOPolicy network + Clarke & Wright + helpers
train.py             # PPO training (logging, checkpointing, resume, plots)
evaluate_ppo.py      # Lightweight evaluation (adaptive model loading)
evaluate.py / evluate.py  # (Interactive / legacy comprehensive evaluation)
units.py             # Domain units/utilities
test_case_evrp.tsp   # Sample EVRP instance (TSP-like format)
best_ppo_policy.pth  # Best validation-reward PPO weights (current run)
trained_ppo_policy.pth  # Final PPO after last episode
ppo_checkpoint.pth   # Latest checkpoint (with optimizer + scheduler)
ppo_training_log.csv # Episode-wise metrics log
best_policy/         # Legacy models (different node counts) – optional
data/                # Generated stats, tours, gifs / plots
README_CLEAN.md      # This documentation
```

---
## 3. State, Action & Reward Design
| Component | Description |
|-----------|-------------|
| State     | Concatenation: one-hot(current_node) + [battery_level] + visited_mask (boolean). Length = 2*N + 1 where N = total nodes incl. depot. |
| Action    | Index of next node (customer / charging station / depot) within N. Mask filters infeasible choices. |
| Value     | Predicted state value (critic). |

### Reward Shaping Terms (arguments in train.py)
| Flag | Meaning | Default |
|------|---------|---------|
| --distance_weight | Scales negative Euclidean step distance | -1.0 |
| --visit_customer_reward | Bonus first time visiting a customer | 0.2 |
| --completion_bonus | Episode success bonus | 2.0 |
| --illegal_penalty | Penalty for infeasible/illegal step | -1.0 |
| --detour_penalty | Penalty for revisiting already visited customer | -0.2 |
| --step_penalty | Mild per-step cost (shorter tours) | -0.01 |

Advantages are computed with gamma≈0.995 and standardized.

---
## 4. PPO Network Architecture
- Dynamic hidden size: hidden = clamp(4 * state_dim, 64..256)
- Layers: Linear → LayerNorm → ReLU → Dropout → Residual Block (Linear + LN + ReLU) → Linear + LN + ReLU → Dropout → Heads (policy & value)
- Regularization: Dropout(0.1), weight decay, gradient clipping (0.5).
- Scheduler: ReduceLROnPlateau (monitors validation reward).
- Early stopping: patience (default 50 episodes without sufficient val improvement).

---
## 5. Training Workflow
1. Random batch of EVRP instances generated each episode (num_samples).
2. Split into train / validation (validation_split).
3. Collect trajectories (masked categorical action sampling).
4. Compute returns + advantages; run multiple PPO epochs (minibatches).
5. Log metrics; save:
  - best_ppo_policy.pth (whenever val improves)
  - ppo_checkpoint.pth (+ optimizer & scheduler state)
  - trained_ppo_policy.pth (final)
6. Append CSV row.
7. Generate plots (reward, distance, customers) at end.

### Resume Training
```powershell
python train.py --resume --checkpoint_path ppo_checkpoint.pth --log_path ppo_training_log.csv
```
Respects prior episodes; continues numbering. Learning rate scheduler & patience restored.

### CSV Columns Explained
| Column | Meaning |
|--------|---------|
| episode | 1-based episode index (continues on resume) |
| train_reward / val_reward | Mean shaped reward across train / val samples |
| loss | policy + value composite (episode average) |
| lr | Current learning rate from scheduler |
| train_distance | Mean total distance (lower is better) |
| train_customers | Mean unique customers visited |
| policy_loss / value_loss | Component losses (averaged) |
| entropy | Last minibatch entropy (higher early = exploration) |
| steps | Total recorded steps (trajectory length) |

---
## 6. Evaluation Modes
### evaluate_ppo.py (Fast)
Loads a saved PPO model AND runs C&W for direct comparison (distance, coverage, efficiency). Includes adaptive loader:
- If model trained on fewer nodes than test case → test case truncated to match model.
- If model expects more nodes than provided → abort with clear message.

### evaluate.py (Full / Interactive)
Optional rich mode: training, GIFs, parameter prompts (legacy / extended experimentation). Use when you need visual artifacts or custom env parameters.

---
## 7. Legacy Models (best_policy/)
Files like policy-TSP20-epoch-189.pt etc. were trained with different node counts. They are NOT directly compatible with the current environment dimensions and will raise size mismatch without adaptation. Keep them only if you: benchmark across sizes, study scaling, or do transfer learning. Otherwise move to an archive.

---
## 8. Adaptive Loading (Dimension Mismatch Fix)
In `evaluate_ppo.py`:
- Inspects saved `state_dict` (fc1.weight width = state_dim, policy_head rows = action_dim).
- Infers original node count: (state_dim - 1)/2.
- If current test has MORE nodes → truncates arrays to pretrained size.
- Otherwise aborts (no unsafe padding performed).

This resolves errors like:
```
size mismatch for fc1.weight: copying param with shape [64, 13] ... got [228, 57]
```

---
## 9. Plots Generated
- ppo_training_curves.png (Train vs Val reward, loss)
- ppo_training_distance_customers.png (Avg distance & customers visited)
Re-run plotting manually by importing pandas/matplotlib and re-reading CSV if needed.

---
## 10. Troubleshooting
| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Size mismatch loading model | Different node count | Use evaluate_ppo.py adaptive loader or match test size |
| Entropy collapses to ~0 early | Over-aggressive exploitation | Increase --entropy_coef (e.g. 0.03) |
| Validation reward stagnant but train rises | Overfitting | Increase validation fraction, reduce ppo_epochs or add early stop |
| Distance not decreasing | Reward weights misaligned | Adjust --distance_weight magnitude |
| Customers not fully visited | Mask / termination | Inspect get_action_mask & ensure completion criteria |
| Learning rate stuck | Plateau not detected | Patience too large or val noisy – smooth or reduce patience |
| Resume ignored | Wrong paths | Ensure --resume AND existing ppo_checkpoint.pth/log CSV present |

---
## 11. Extending / Next Ideas
- Add seeds for reproducibility (torch.manual_seed, numpy.random.seed).
- Curriculum scaling (already partial) – formalize progressive node count schedule.
- Multi-agent vehicles (extend action semantics).
- Battery-aware reward shaping (explicit energy cost penalty).
- Export ONNX for inference.

---
## 12. License & Attribution
This repository integrates a classical heuristic (Clarke & Wright) and a custom PPO implementation. Cite appropriately if used in academic work.

---
## 13. Minimal Concept Recap
Input: randomly generated EVRP instances (positions + one charging station).  
Policy: chooses next node until termination.  
Objective: maximize shaped reward (equivalently reduce distance & penalties while covering customers).  
Baseline: C&W provides deterministic heuristic baseline for distance & coverage.

---
## 14. Example End-to-End Session
```powershell
# Train 30 episodes, save best model
python train.py --num_nodes 8 --num_samples 64 --num_episodes 30

# Resume for 20 more
python train.py --resume --num_nodes 8 --num_samples 64 --num_episodes 50

# Evaluate
python evaluate_ppo.py --model_path best_ppo_policy.pth
```

---
## 15. Current Status
✅ Training stable  
✅ Checkpoint + resume working  
✅ Adaptive evaluation solves dimension mismatches  
✅ Comparative baseline included  
✅ Logs + plots reproducible  

Ready for experimentation and extension. 🚀

---
If anything is unclear, open an issue or inspect the referenced file in this README.
