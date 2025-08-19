import torch
import numpy as np
import argparse
import os
from mp_utils import create_tour, EVEnvironment, swap_2opt, PPOPolicy
import json

def train_ppo(
    num_episodes,
    num_nodes,
    num_samples,
    validation_split=0.2,
    lr=3e-4,
    entropy_coef=0.02,
    ppo_epochs=4,
    minibatch_size=256,
    # Reward shaping weights
    distance_weight=-1.0,
    visit_customer_reward=0.2,
    completion_bonus=2.0,
    illegal_penalty=-1.0,
    detour_penalty=-0.2,
    step_penalty=-0.01,
    resume=False,
    checkpoint_path="ppo_checkpoint.pth",
    log_path="ppo_training_log.csv",
    debug=False,
):
    """
    Train PPO policy for EVRP with improved regularization and validation.

    :param int num_episodes: Number of training episodes
    :param int num_nodes: Number of customer nodes
    :param int num_samples: Number of samples per episode
    :param float validation_split: Fraction of data to use for validation
    """
    # Initialize PPO policy
    total_nodes = num_nodes + 1  # Including depot
    state_dim = total_nodes + 1 + total_nodes  # one_hot + battery + visited_mask
    action_dim = total_nodes
    policy = PPOPolicy(state_dim, action_dim, dropout_rate=0.1)
    print(f"[DEBUG][init] policy id: {id(policy)}, type: {type(policy)}")
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr, weight_decay=1e-4)
    print(f"[DEBUG][init] optimizer id: {id(optimizer)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.8, patience=20, min_lr=1e-6
    )
    
    # Early stopping parameters
    best_val_reward = -float('inf')
    patience = 50
    patience_counter = 0
    
    # Training history
    train_rewards = []
    val_rewards = []
    train_losses = []

    # Prepare CSV logging
    import csv
    csv_log_path = log_path
    csv_exists = os.path.exists(csv_log_path)
    # If an existing log lacks header, insert one so pandas/DictReader parse correctly
    if csv_exists:
        try:
            with open(csv_log_path, 'r', newline='') as f_chk:
                first_line = f_chk.readline().strip()
            expected_prefix = 'episode'
            if first_line and not first_line.lower().startswith(expected_prefix):
                with open(csv_log_path, 'r', newline='') as f_in:
                    lines = f_in.read().strip().splitlines()
                header = ["episode", "train_reward", "val_reward", "loss", "lr", "train_distance", "train_customers", "policy_loss", "value_loss", "entropy", "steps"]
                with open(csv_log_path, 'w', newline='') as f_out:
                    f_out.write(','.join(header) + "\n")
                    for ln in lines:
                        if ln.strip():
                            f_out.write(ln + "\n")
                print(f"[LOG][HEADER] Added missing header to existing log '{csv_log_path}'")
        except Exception as e:
            print(f"[LOG][HEADER][WARN] Could not repair log header: {e}")
    # Load history if resuming
    if resume and csv_exists:
        with open(csv_log_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    train_rewards.append(float(row["train_reward"]))
                    val_rewards.append(float(row["val_reward"]))
                    train_losses.append(float(row["loss"]))
                except Exception:
                    pass
            last_episode_completed = len(train_rewards)
    else:
        last_episode_completed = 0
    csv_file = open(csv_log_path, mode=("a" if resume and csv_exists else "w"), newline="")
    csv_writer = csv.writer(csv_file)
    if not (resume and csv_exists):
        csv_writer.writerow(["episode", "train_reward", "val_reward", "loss", "lr", "train_distance", "train_customers", "policy_loss", "value_loss", "entropy", "steps"])

    # Resume from checkpoint if requested
    if resume and os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            policy.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            best_val_reward = ckpt.get("best_val_reward", best_val_reward)
            patience_counter = ckpt.get("patience_counter", patience_counter)
            print(f"[RESUME] Loaded checkpoint from {checkpoint_path} (episode {ckpt.get('episode', '?')})")
        except Exception as e:
            print(f"[RESUME][WARNING] Failed to load checkpoint: {e}")
    
    print(f"[INFO] Training PPO on device: {device}")
    print(f"[INFO] Model parameters: {sum(p.numel() for p in policy.parameters() if p.requires_grad):,}")
    
    early_stop = False
    for episode in range(last_episode_completed, num_episodes):
        if early_stop:
            break
        policy.train()  # Ensure model is in training mode for each episode
        print(f"\n{'='*40}\n[EPISODE {episode+1}/{num_episodes}] Starting training...")
        # Generate random EVRP instances
        positions = np.random.rand(num_samples, num_nodes, 2)  # Customer nodes
        charging_station_pos = np.array([[0.5, 0.5]])  # Charging station
        all_positions = [np.vstack([pos, charging_station_pos]) for pos in positions]
        envs = [EVEnvironment(pos, [num_nodes], battery_capacity=150, consumption_rate=0.15, min_soc=0.1, charging_time=0.5) for pos in all_positions]

        # Split into train and validation
        split_idx = int(num_samples * (1 - validation_split))
        train_envs = envs[:split_idx]
        val_envs = envs[split_idx:]

        # Training phase
        train_rewards_episode = []  # per-sample episode cumulative reward
        train_distances_episode = []  # per-sample total distance
        train_customers_episode = []  # per-sample customers visited
        train_log_probs = []        # flattened per-step log probs
        train_values = []           # flattened per-step state values
        train_states = []           # flattened per-step states
        train_actions = []          # flattened per-step actions
        train_masks = []            # flattened per-step masks
        step_rewards = []           # flattened per-step shaped rewards (for returns)
        step_entropies = []         # per-step entropies for diagnostics

        for sample in range(len(train_envs)):
            env = train_envs[sample]
            reset_state = env.reset()
            if debug:
                print(f"[DEBUG][env.reset] reset_state: {reset_state}")
            if reset_state is not None:
                if isinstance(reset_state, np.ndarray):
                    if debug:
                        print(f"[DEBUG][env.reset] reset_state.shape: {reset_state.shape}")
                elif isinstance(reset_state, tuple):
                    if debug:
                        print(f"[DEBUG][env.reset] reset_state (tuple lens): {[type(x) for x in reset_state]}, {[np.shape(x) if hasattr(x, 'shape') else type(x) for x in reset_state]}")
            # Define current_node and initialize state from reset_state
            if isinstance(reset_state, tuple):
                tour, soc, current_pos, visited = reset_state
                visited_mask = np.array(visited)
                battery = float(soc.magnitude) if hasattr(soc, 'magnitude') else float(soc)
                current_node_one_hot = np.zeros(total_nodes)
                current_node_one_hot[current_pos] = 1.0
                state = np.concatenate([current_node_one_hot, [battery], visited_mask])
                current_node = current_pos
            else:
                state = reset_state
                current_node = env.current_node if hasattr(env, 'current_node') else 0

            episode_reward = 0
            episode_distance = 0
            episode_customers = set()
            episode_log_probs = []
            episode_values = []
            episode_states = []
            episode_actions = []
            episode_masks = []
            action_trace = []  # for debug on first sample

            done = False
            steps = 0
            max_steps = 4 * num_nodes + 20

            while not done and steps < max_steps:
                # Get action mask
                mask = env.get_action_mask()
                mask = np.asarray(mask).flatten()

                # Debug: print state shape and assert
                if debug:
                    print(f"[DEBUG][env step] state.shape: {state.shape}, expected: {state_dim}, state: {state}")
                assert state.shape[0] == state_dim, f"State shape mismatch: got {state.shape[0]}, expected {state_dim}"
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                mask_tensor = torch.FloatTensor(mask).unsqueeze(0).to(device)
                probs, value = policy(state_tensor, action_mask=mask_tensor)

                dist = torch.distributions.Categorical(probs)
                action = dist.sample().item()
                log_prob = dist.log_prob(torch.tensor([action], device=device, dtype=torch.long))
                # Track entropy per step for diagnostics
                try:
                    step_entropies.append(dist.entropy().mean().item())
                except Exception:
                    pass

                # Take raw environment step
                # Record visitation status BEFORE step so we can count first visits correctly
                pre_step_visited = False
                if hasattr(env, 'visited') and action < len(env.visited):
                    pre_step_visited = env.visited[action]
                next_state, reward, done, info = env.step(action)

                # Custom reward shaping (environment raw reward assumed negative distance / cost)
                shaped_reward = 0.0
                # Distance penalty if positions available
                if hasattr(env, 'positions') and isinstance(next_state, tuple):
                    prev_node_idx = current_node
                    if action < len(env.positions) and prev_node_idx < len(env.positions):
                        dist = float(np.linalg.norm(env.positions[prev_node_idx] - env.positions[action]))
                        shaped_reward += distance_weight * dist
                    if debug:
                        print(f"[DEBUG][env.step] next_state: {next_state}")
                        if isinstance(next_state, np.ndarray):
                            print(f"[DEBUG][env.step] next_state.shape: {next_state.shape}")
                        elif isinstance(next_state, tuple):
                            print(f"[DEBUG][env.step] next_state (tuple lens): {[type(x) for x in next_state]}, {[np.shape(x) if hasattr(x, 'shape') else type(x) for x in next_state]}")
                # Reward first time visiting a new customer (exclude depot assumed 0)
                if hasattr(env, 'visited') and action < len(env.visited):
                    if not env.visited[action] and action != 0:
                        shaped_reward += visit_customer_reward
                    else:
                        # mild detour penalty when revisiting
                        shaped_reward += detour_penalty
                # Penalize illegal/infeasible moves
                if isinstance(info, dict) and 'status' in info:
                    if any(k in str(info['status']).lower() for k in ['illegal', 'infeasible']):
                        shaped_reward += illegal_penalty
                # Step penalty to encourage shorter tours
                shaped_reward += step_penalty
                # Completion bonus if episode ends without illegal status and all customers visited (heuristic)
                if done and isinstance(info, dict) and not any(k in str(info.get('status','')).lower() for k in ['illegal','infeasible']):
                    shaped_reward += completion_bonus
                # If environment provides shaped_reward, optionally blend (50%)
                if isinstance(info, dict) and 'shaped_reward' in info:
                    shaped_reward = 0.5 * shaped_reward + 0.5 * float(info['shaped_reward'])


                # Track distance if possible
                if hasattr(env, 'positions') and current_node < len(env.positions) and action < len(env.positions):
                    dist_val = float(np.linalg.norm(env.positions[current_node] - env.positions[action]))
                    episode_distance += dist_val

                # Track customers visited (excluding depot=0)
                if action != 0 and hasattr(env, 'visited') and action < len(env.visited):
                    if not pre_step_visited:  # counted only on first visit
                        episode_customers.add(action)

                # Store experience

                episode_log_probs.append(log_prob)
                # Debug: print value.requires_grad when storing
                if debug:
                    print(f"[DEBUG] (rollout) value.requires_grad: {value.requires_grad}, value.grad_fn: {value.grad_fn}")
                episode_values.append(value)
                episode_states.append(state)
                episode_actions.append(action)
                episode_masks.append(mask)
                step_rewards.append(shaped_reward)
                action_trace.append(action)

                episode_reward += shaped_reward
                steps += 1

                # Update state - handle the environment's state format
                if isinstance(next_state, tuple):
                    # Environment returns (tour, soc, current_pos, visited)
                    tour, soc, current_pos, visited = next_state
                    visited_mask = np.array(visited)
                    battery = float(soc.magnitude) if hasattr(soc, 'magnitude') else float(soc)
                    current_node_one_hot = np.zeros(total_nodes)
                    current_node_one_hot[current_pos] = 1.0
                    state = np.concatenate([current_node_one_hot, [battery], visited_mask])
                    if debug:
                        print(f"[DEBUG][state update] state.shape: {state.shape}, expected: {state_dim}, state: {state}")
                    assert state.shape[0] == state_dim, f"State shape mismatch after update: got {state.shape[0]}, expected {state_dim}"
                    current_node = current_pos
                else:
                    state = next_state

                if done:
                    break

            if debug and sample == 0 and episode % 10 == 0:
                unique_actions = len(set(action_trace))
                print(f"[TRACE] Sample0 Episode {episode+1}: actions={action_trace} unique={unique_actions} len={len(action_trace)}")

            train_rewards_episode.append(episode_reward)
            train_distances_episode.append(episode_distance)
            train_customers_episode.append(len(episode_customers))
            train_log_probs.extend(episode_log_probs)
            train_values.extend(episode_values)
            train_states.extend(episode_states)
            train_actions.extend(episode_actions)
            train_masks.extend(episode_masks)

        # Aggregate metrics for this episode (after iterating all training samples)
        train_distance_avg = np.mean(train_distances_episode) if train_distances_episode else 0.0
        train_customers_avg = np.mean(train_customers_episode) if train_customers_episode else 0.0
        if step_entropies:
            avg_sampling_entropy = float(np.mean(step_entropies))
        else:
            avg_sampling_entropy = float('nan')

        # Validation phase
        val_rewards_episode = []
        policy.eval()  # Set to evaluation mode
        with torch.no_grad():
            for sample in range(len(val_envs)):
                env = val_envs[sample]
                reset_state = env.reset()
                if debug:
                    print(f"[DEBUG][env.reset] reset_state: {reset_state}")
                if reset_state is not None:
                    if isinstance(reset_state, np.ndarray):
                        if debug:
                            print(f"[DEBUG][env.reset] reset_state.shape: {reset_state.shape}")
                    elif isinstance(reset_state, tuple):
                        if debug:
                            print(f"[DEBUG][env.reset] reset_state (tuple lens): {[type(x) for x in reset_state]}, {[np.shape(x) if hasattr(x, 'shape') else type(x) for x in reset_state]}")

                # Get initial state
                policy.eval()  # Set to evaluation mode
                visited_mask = np.array(env.visited)
                battery = env.battery
                current_node_one_hot = np.zeros(total_nodes)
                current_node_one_hot[current_node] = 1.0
                state = np.concatenate([current_node_one_hot, [battery], visited_mask])

                episode_reward = 0
                done = False
                steps = 0
                max_steps = 4 * num_nodes + 20

                while not done and steps < max_steps:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    mask = env.get_action_mask()
                    mask = np.asarray(mask).flatten()
                    mask_tensor = torch.FloatTensor(mask).unsqueeze(0).to(device)
                    probs, value = policy(state_tensor, action_mask=mask_tensor)

                    # Use argmax for deterministic validation
                    action = torch.argmax(probs, dim=-1).item()

                    # Take step in environment
                    next_state, reward, done, info = env.step(action)
                    if debug:
                        print(f"[DEBUG][val env.step] next_state: {next_state}")
                        if isinstance(next_state, np.ndarray):
                            print(f"[DEBUG][val env.step] next_state.shape: {next_state.shape}")
                        elif isinstance(next_state, tuple):
                            print(f"[DEBUG][val env.step] next_state (tuple lens): {[type(x) for x in next_state]}, {[np.shape(x) if hasattr(x, 'shape') else type(x) for x in next_state]}")

                    episode_reward += reward
                    steps += 1

                    # Update state - handle the environment's state format
                    if isinstance(next_state, tuple):
                        # Environment returns (tour, soc, current_pos, visited)
                        try:
                            tour, soc, current_pos, visited = next_state
                            visited_mask = np.array(visited)
                            battery = float(soc.magnitude) if hasattr(soc, 'magnitude') else float(soc)
                            current_node_one_hot = np.zeros(total_nodes)
                            current_node_one_hot[current_pos] = 1.0
                            state = np.concatenate([current_node_one_hot, [battery], visited_mask])
                            if debug:
                                print(f"[DEBUG][state update] state.shape: {state.shape}, expected: {state_dim}, state: {state}")
                            assert state.shape[0] == state_dim, f"State shape mismatch after update: got {state.shape[0]}, expected {state_dim}"
                            current_node = current_pos
                        except Exception as e:
                            if debug:
                                print(f"[DEBUG][state update] Exception: {e}, next_state: {next_state}")
                            state = None
                    elif next_state is not None:
                        state = next_state
                    else:
                        state = None
                    if state is not None:
                        if debug:
                            if hasattr(state, 'shape'):
                                print(f"[DEBUG][env step] state.shape: {state.shape}, expected: {state_dim}, state: {state}")
                            else:
                                print(f"[DEBUG][env step] state has no 'shape' attribute, type: {type(state)}, value: {state}")
                    else:
                        if debug:
                            print(f"[DEBUG][env step] state is None after step! next_state: {next_state}")
                        break

            val_rewards_episode.append(episode_reward)
        # END validation no_grad block
        policy.train()  # Set back to training mode

        # Update policy with PPO (now OUTSIDE no_grad)
        if train_log_probs and train_values:
            # Convert to tensors
            log_probs = torch.stack(train_log_probs).to(device)
            values = torch.cat(train_values).squeeze(-1).to(device)
            states_tensor = torch.tensor(np.array(train_states), dtype=torch.float32, device=device)
            if debug:
                print(f"[DEBUG][PPO update] states_tensor.shape: {states_tensor.shape}, expected: (?, {state_dim})")
            assert states_tensor.shape[1] == state_dim, f"PPO update states_tensor shape mismatch: got {states_tensor.shape[1]}, expected {state_dim}"
            actions_tensor = torch.tensor(train_actions, dtype=torch.long, device=device)
            masks_tensor = torch.tensor(np.array(train_masks), dtype=torch.float32, device=device)
            if debug:
                print('[DEBUG] policy.training (before PPO update):', policy.training)
                print(f"[DEBUG][PPO update] policy id: {id(policy)}, type: {type(policy)}")
                print(f"[DEBUG][PPO update] optimizer id: {id(optimizer)}")

            # Use collected per-step rewards
            step_rewards_flat = step_rewards
            gamma = 0.995
            returns = []
            G = 0.0
            for r in reversed(step_rewards_flat):
                G = r + gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns, dtype=torch.float32, device=device)
            if returns.shape[0] != values.shape[0]:
                min_len = min(returns.shape[0], values.shape[0])
                returns = returns[:min_len]
                values = values[:min_len]
                log_probs = log_probs[:min_len]
                states_tensor = states_tensor[:min_len]
                actions_tensor = actions_tensor[:min_len]
                masks_tensor = masks_tensor[:min_len]
            advantages = returns - values.detach()
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            clip_epsilon = 0.1
            update_epochs = ppo_epochs
            total_policy_loss = 0.0
            total_value_loss = 0.0
            n_steps = states_tensor.shape[0]
            indices = np.arange(n_steps)
            for _ in range(update_epochs):
                np.random.shuffle(indices)
                for start in range(0, n_steps, minibatch_size):
                    end = start + minibatch_size
                    mb_idx = indices[start:end]
                    mb_states = states_tensor[mb_idx].clone().requires_grad_(True)
                    mb_actions = actions_tensor[mb_idx]
                    mb_old_log_probs = log_probs.detach()[mb_idx]
                    mb_adv = advantages[mb_idx]
                    mb_returns = returns[mb_idx]
                    mb_masks = masks_tensor[mb_idx]
                    probs, value = policy(mb_states, action_mask=mb_masks)
                    dist = torch.distributions.Categorical(probs)
                    new_log_probs = dist.log_prob(mb_actions)
                    entropy = dist.entropy().mean()
                    ratio = (new_log_probs - mb_old_log_probs).exp()
                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * mb_adv
                    policy_loss = -torch.min(surr1, surr2).mean() - entropy_coef * entropy
                    value_loss = (mb_returns - value.squeeze(-1)).pow(2).mean()
                    total_loss = policy_loss + 0.5 * value_loss
                    optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
                    optimizer.step()
                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()

            denom = max(1, update_epochs * (n_steps // max(1, minibatch_size)))
            avg_policy_loss = total_policy_loss / denom
            avg_value_loss = total_value_loss / denom
            train_reward_avg = np.mean(train_rewards_episode)
            val_reward_avg = np.mean(val_rewards_episode) if val_rewards_episode else -float('inf')
            train_rewards.append(train_reward_avg)
            val_rewards.append(val_reward_avg)
            train_losses.append(avg_policy_loss + avg_value_loss)
            scheduler.step(val_reward_avg)
            if val_reward_avg > best_val_reward + 0.01:
                best_val_reward = val_reward_avg
                patience_counter = 0
                torch.save(policy.state_dict(), "best_ppo_policy.pth")
                print(f"[HIGHLIGHT] New best validation reward: {best_val_reward:.2f} (model saved)")
            else:
                patience_counter += 1
            # Use average sampling entropy collected during rollout (better episode-level signal)
            avg_entropy = float(avg_sampling_entropy) if not np.isnan(avg_sampling_entropy) else (float(entropy.item()) if 'entropy' in locals() else float('nan'))
            csv_writer.writerow([episode+1, train_reward_avg, val_reward_avg, avg_policy_loss + avg_value_loss, optimizer.param_groups[0]['lr'], train_distance_avg, train_customers_avg, avg_policy_loss, avg_value_loss, avg_entropy, len(step_rewards_flat)])
            csv_file.flush()

            # Save checkpoint each episode
            ckpt = {
                "episode": episode+1,
                "model_state": policy.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val_reward": best_val_reward,
                "patience_counter": patience_counter,
                "config": {
                    "num_nodes": num_nodes,
                    "resume": resume,
                    "distance_weight": distance_weight,
                    "visit_customer_reward": visit_customer_reward,
                }
            }
            torch.save(ckpt, checkpoint_path)
            # Also save lightweight JSON metadata
            with open(checkpoint_path + ".meta.json", "w") as mf:
                json.dump({k: v for k, v in ckpt.items() if k not in ["model_state", "optimizer_state", "scheduler_state"]}, mf, indent=2)
            print(f"[LOG] Ep {episode+1}/{num_episodes} | TrainR {train_reward_avg:.2f} | ValR {val_reward_avg:.2f} | Loss {avg_policy_loss + avg_value_loss:.4f} | VLoss {avg_value_loss:.4f} | PLoss {avg_policy_loss:.4f} | Steps {len(step_rewards_flat)} | LR {optimizer.param_groups[0]['lr']:.2e}")
            if patience_counter >= patience:
                print(f"[HIGHLIGHT] Early stopping triggered after {episode+1} episodes")
        else:
            print(f"[WARNING] Episode {episode+1}, No valid actions taken")
        # Reset per-episode storage lists for next episode
        train_log_probs.clear(); train_values.clear(); train_states.clear(); train_actions.clear(); train_masks.clear(); step_rewards.clear(); step_entropies.clear()
        if patience_counter >= patience:
            print("[INFO] Early stopping condition met — exiting training loop.")
            early_stop = True
            # do not break here; let outer episode loop condition handle
            
    
    # Save the final trained model
    torch.save(policy.state_dict(), "trained_ppo_policy.pth")
    print("[INFO] Trained PPO model saved as 'trained_ppo_policy.pth'")

    # Close CSV file
    csv_file.close()
    print(f"[INFO] Training log saved to '{csv_log_path}'")

    # Plot training curves
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot rewards
    ax1.plot(train_rewards, label='Train Reward', alpha=0.7)
    ax1.plot(val_rewards, label='Validation Reward', alpha=0.7)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('PPO Training and Validation Rewards')
    ax1.legend()
    ax1.grid(True)

    # Plot loss
    ax2.plot(train_losses, label='Training Loss', color='red', alpha=0.7)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.set_title('PPO Training Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    # Extra plots: distance & customers
    try:
        import pandas as pd
        df_log = pd.read_csv(csv_log_path)
        fig2, (bx1, bx2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        if 'train_distance' in df_log.columns:
            bx1.plot(df_log['episode'], df_log['train_distance'], label='Avg Train Distance', color='purple')
            bx1.set_ylabel('Distance')
            bx1.legend(); bx1.grid(True)
        if 'train_customers' in df_log.columns:
            bx2.plot(df_log['episode'], df_log['train_customers'], label='Avg Customers Visited', color='teal')
            bx2.set_ylabel('Customers')
            bx2.set_xlabel('Episode')
            bx2.legend(); bx2.grid(True)
        plt.tight_layout()
        plt.savefig('ppo_training_distance_customers.png', dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"[PLOT][WARNING] Could not generate extra plots: {e}")

    plt.savefig('ppo_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

    return policy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO policy for EVRP")
    parser.add_argument("--num_nodes", type=int, default=5, help="Number of customer nodes")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples per episode")
    parser.add_argument("--num_episodes", type=int, default=200, help="Number of training episodes")
    parser.add_argument("--validation_split", type=float, default=0.2, help="Fraction of data for validation")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--entropy_coef", type=float, default=0.02, help="Entropy coefficient for exploration (lower encourages exploitation)")
    parser.add_argument("--ppo_epochs", type=int, default=4, help="Number of PPO optimization epochs per update")
    parser.add_argument("--minibatch_size", type=int, default=256, help="Minibatch size for PPO updates")
    parser.add_argument("--distance_weight", type=float, default=-1.0, help="Weight for distance penalty in shaped reward")
    parser.add_argument("--visit_customer_reward", type=float, default=0.2, help="Reward for visiting a new customer")
    parser.add_argument("--completion_bonus", type=float, default=2.0, help="Bonus when route/episode completes successfully")
    parser.add_argument("--illegal_penalty", type=float, default=-1.0, help="Penalty for illegal/infeasible move")
    parser.add_argument("--detour_penalty", type=float, default=-0.2, help="Penalty for revisiting an already visited node (detour)")
    parser.add_argument("--step_penalty", type=float, default=-0.01, help="Per-step penalty to encourage shorter tours")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint and log")
    parser.add_argument("--checkpoint_path", type=str, default="ppo_checkpoint.pth", help="Path for saving/loading checkpoint")
    parser.add_argument("--log_path", type=str, default="ppo_training_log.csv", help="CSV log file path")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug prints")
    args = parser.parse_args()
    train_ppo(
        args.num_episodes,
        args.num_nodes,
        args.num_samples,
        args.validation_split,
        lr=args.lr,
        entropy_coef=args.entropy_coef,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        distance_weight=args.distance_weight,
        visit_customer_reward=args.visit_customer_reward,
        completion_bonus=args.completion_bonus,
        illegal_penalty=args.illegal_penalty,
        detour_penalty=args.detour_penalty,
        step_penalty=args.step_penalty,
    resume=args.resume,
    checkpoint_path=args.checkpoint_path,
    log_path=args.log_path,
    debug=args.debug,
    )