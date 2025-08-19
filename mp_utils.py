# --- Clarke & Wright's Savings Algorithm for EVRP ---
def clarke_wright_evrp(positions, demands, depot, charging_stations, vehicle_capacity, battery_capacity, consumption_rate, min_soc, charging_time, max_vehicles=None):
    """
    Clarke and Wright's Savings Algorithm adapted for EVRP.
    Returns a list of feasible routes (each a list of node indices, starting and ending at depot, with charging stations as needed).
    If max_vehicles is set, tries to merge routes to not exceed this limit.
    Includes debug prints for merge failures.
    """
    # Ensure vehicle_capacity and demands are floats (kilogram)
    vehicle_capacity_val = vehicle_capacity.magnitude if hasattr(vehicle_capacity, 'magnitude') else float(vehicle_capacity)
    demands_val = [d.magnitude if hasattr(d, 'magnitude') else float(d) for d in demands]
    # Ensure consumption_rate is float (watt_hour per kilometer)
    if hasattr(consumption_rate, 'to'):
        consumption_rate_val = consumption_rate.to('watt_hour/kilometer').magnitude
    else:
        consumption_rate_val = float(consumption_rate)
    customers = [i for i in range(len(demands_val)) if i != depot and i not in charging_stations and demands_val[i] > 0]
    dist = np.linalg.norm(positions[:, np.newaxis] - positions, axis=2)  # in kilometers (float)
    routes = {i: [depot, i, depot] for i in customers}
    route_loads = {i: demands_val[i] for i in customers}
    import itertools
    savings = []
    for i, j in itertools.combinations(customers, 2):
        s = dist[depot][i] + dist[depot][j] - dist[i][j]
        savings.append(((i, j), s))
    savings.sort(key=lambda x: -x[1])
    def is_battery_feasible(route):
        soc = battery_capacity.to('watt_hour').magnitude if hasattr(battery_capacity, 'to') else float(battery_capacity)
        min_soc_energy = min_soc * (battery_capacity.to('watt_hour').magnitude if hasattr(battery_capacity, 'to') else float(battery_capacity))
        for idx in range(1, len(route)):
            prev, curr = route[idx-1], route[idx]
            d_km = float(dist[prev][curr])  # kilometers
            consumption_wh = d_km * consumption_rate_val  # watt_hour
            soc_check = soc - consumption_wh
            if soc_check < min_soc_energy:
                inserted = False
                for cs in charging_stations:
                    d1_km = float(dist[prev][cs])
                    d2_km = float(dist[cs][curr])
                    c1_wh = d1_km * consumption_rate_val
                    c2_wh = d2_km * consumption_rate_val
                    soc_c1 = soc - c1_wh
                    batt_c2 = (battery_capacity.to('watt_hour').magnitude if hasattr(battery_capacity, 'to') else float(battery_capacity)) - c2_wh
                    if soc_c1 >= min_soc_energy and batt_c2 >= min_soc_energy:
                        route = route[:idx] + [cs] + route[idx:]
                        soc = (battery_capacity.to('watt_hour').magnitude if hasattr(battery_capacity, 'to') else float(battery_capacity)) - c2_wh
                        inserted = True
                        break
                if not inserted:
                    return False, route
            else:
                soc = soc - consumption_wh
                if curr in charging_stations:
                    soc = battery_capacity.to('watt_hour').magnitude if hasattr(battery_capacity, 'to') else float(battery_capacity)
        return True, route
    for (i, j), s in savings:
        route_i = None
        route_j = None
        for key, r in routes.items():
            if r[1] == i and r[-2] == i:
                route_i = key
            if r[1] == j and r[-2] == j:
                route_j = key
        if route_i is not None and route_j is not None and route_i != route_j:
            r_i = routes[route_i]
            r_j = routes[route_j]
            if r_i[-2] == i and r_j[1] == j:
                total_load = route_loads[route_i] + route_loads[route_j]
                if total_load <= vehicle_capacity_val:
                    merged = r_i[:-1] + r_j[1:]
                    feasible, merged_with_charging = is_battery_feasible(merged)
                    if feasible:
                        routes[route_i] = merged_with_charging
                        route_loads[route_i] = total_load
                        del routes[route_j]
                        del route_loads[route_j]
                    else:
                        print(f"DEBUG: Could not merge routes {r_i} and {r_j} due to battery/charging constraint.")
                else:
                    print(f"DEBUG: Could not merge routes {r_i} and {r_j} due to vehicle capacity constraint.")
    # Step 5: Enforce max_vehicles constraint if set
    if max_vehicles is not None and len(routes) > max_vehicles:
        import heapq
        warning_printed = False
        while len(routes) > max_vehicles:
            route_keys = list(routes.keys())
            merged_any = False
            route_keys_sorted = sorted(route_keys, key=lambda k: route_loads[k])
            for i in range(len(route_keys_sorted)):
                for j in range(i+1, len(route_keys_sorted)):
                    k1, k2 = route_keys_sorted[i], route_keys_sorted[j]
                    r1, r2 = routes[k1], routes[k2]
                    total_load = route_loads[k1] + route_loads[k2]
                    if total_load <= vehicle_capacity_val:
                        merged = r1[:-1] + r2[1:]
                        feasible, merged_with_charging = is_battery_feasible(merged)
                        if feasible:
                            routes[k1] = merged_with_charging
                            route_loads[k1] = total_load
                            del routes[k2]
                            del route_loads[k2]
                            merged_any = True
                            break
                        else:
                            print(f"DEBUG: Post-processing: Could not merge {r1} and {r2} due to battery/charging constraint.")
                    else:
                        print(f"DEBUG: Post-processing: Could not merge {r1} and {r2} due to vehicle capacity constraint.")
                if merged_any:
                    break
            if not merged_any:
                if not warning_printed:
                    print(f"WARNING: Could not merge routes to satisfy max_vehicles={max_vehicles}. Some customers may not be served.")
                    warning_printed = True
                break
        # STRICT: If still too many routes, do NOT add more routes. Return only the first max_vehicles routes.
        if len(routes) > max_vehicles:
            sorted_routes = sorted(routes.items(), key=lambda x: route_loads[x[0]], reverse=True)
            routes = dict(sorted_routes[:max_vehicles])
            # Mark that not all customers are served
            strict_vehicle_limit = True
        else:
            strict_vehicle_limit = False
    else:
        strict_vehicle_limit = False
    # Post-processing: Try to merge any remaining small routes if possible
    route_keys = list(routes.keys())
    merged_any = True
    while merged_any:
        merged_any = False
        route_keys = list(routes.keys())
        for i in range(len(route_keys)):
            for j in range(i+1, len(route_keys)):
                k1, k2 = route_keys[i], route_keys[j]
                r1, r2 = routes[k1], routes[k2]
                total_load = route_loads[k1] + route_loads[k2]
                if total_load <= vehicle_capacity_val:
                    merged = r1[:-1] + r2[1:]
                    feasible, merged_with_charging = is_battery_feasible(merged)
                    if feasible:
                        print(f"DEBUG: Post-processing: Successfully merged {r1} and {r2} into {merged_with_charging}")
                        routes[k1] = merged_with_charging
                        route_loads[k1] = total_load
                        del routes[k2]
                        del route_loads[k2]
                        merged_any = True
                        break
                    else:
                        print(f"DEBUG: Post-processing: Could not merge {r1} and {r2} due to battery/charging constraint.")
                else:
                    print(f"DEBUG: Post-processing: Could not merge {r1} and {r2} due to vehicle capacity constraint.")
            if merged_any:
                break
    # --- NEW STEP: Try to serve any leftover customer with a single-customer route ---
    # Only do this if strict_vehicle_limit is False
    if not strict_vehicle_limit:
        # Find all customers with positive demand
        all_customers = set(i for i in range(len(demands)) if i != depot and i not in charging_stations and demands[i] > 0)
        # Find all customers already included in routes
        served_customers = set()
        for r in routes.values():
            for node in r:
                if node in all_customers:
                    served_customers.add(node)
        # For each unserved customer, try to create a single-customer route
        for cust in all_customers - served_customers:
            route = [depot, cust, depot]
            if demands_val[cust] <= vehicle_capacity_val:
                feasible, route_with_charging = is_battery_feasible(route)
                if feasible:
                    print(f"DEBUG: Added single-customer route for leftover customer {cust}: {route_with_charging}")
                    routes[cust] = route_with_charging
                    route_loads[cust] = demands[cust]
                else:
                    print(f"DEBUG: Could not serve leftover customer {cust} even with single-customer route (battery/charging constraint).")
            else:
                print(f"DEBUG: Could not serve leftover customer {cust} (demand exceeds vehicle capacity).")
    return list(routes.values()), strict_vehicle_limit
# Ensure os is imported for file operations
import os
import torch
import torch.nn as nn
import torch.optim as optim

# PPO Policy and PPO EVRP logic moved from evluate.py
class PPOPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, dropout_rate=0.1):
        super().__init__()
        # Adaptive hidden dimensions based on state size
        hidden_dim = max(64, min(256, state_dim * 4))
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Deeper network with residual connections
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # Layer normalization for better training stability (works with any batch size)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim // 2)
        
        self.policy_head = nn.Linear(hidden_dim // 2, action_dim)
        self.value_head = nn.Linear(hidden_dim // 2, 1)
        
        # Initialize weights properly
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x, action_mask=None):
        with torch.enable_grad():
            print(f"[DEBUG][PPOPolicy.forward][input] x.requires_grad: {x.requires_grad}, x.grad_fn: {x.grad_fn}, x.shape: {x.shape}, x.dtype: {x.dtype}")
            # Handle both single samples and batches
            if len(x.shape) == 1:
                x = x.unsqueeze(0)

            print(f"[DEBUG][PPOPolicy.forward] self.training: {self.training}")
            print(f"[DEBUG][PPOPolicy.forward] fc1.weight.requires_grad: {self.fc1.weight.requires_grad}")
            x = self.fc1(x)
            print(f"[DEBUG][PPOPolicy.forward][after fc1] x.requires_grad: {x.requires_grad}, x.grad_fn: {x.grad_fn}")
            x = self.ln1(x)
            print(f"[DEBUG][PPOPolicy.forward][after ln1] x.requires_grad: {x.requires_grad}, x.grad_fn: {x.grad_fn}")
            x = torch.relu(x)
            print(f"[DEBUG][PPOPolicy.forward][after relu1] x.requires_grad: {x.requires_grad}, x.grad_fn: {x.grad_fn}")
            x = self.dropout(x)
            print(f"[DEBUG][PPOPolicy.forward][after dropout1] x.requires_grad: {x.requires_grad}, x.grad_fn: {x.grad_fn}")

            # Second layer with residual connection
            residual = x
            x = self.fc2(x)
            print(f"[DEBUG][PPOPolicy.forward][after fc2] x.requires_grad: {x.requires_grad}, x.grad_fn: {x.grad_fn}")
            x = self.ln2(x)
            print(f"[DEBUG][PPOPolicy.forward][after ln2] x.requires_grad: {x.requires_grad}, x.grad_fn: {x.grad_fn}")
            x = torch.relu(x + residual)  # Residual connection
            print(f"[DEBUG][PPOPolicy.forward][after relu2] x.requires_grad: {x.requires_grad}, x.grad_fn: {x.grad_fn}")
            x = self.dropout(x)
            print(f"[DEBUG][PPOPolicy.forward][after dropout2] x.requires_grad: {x.requires_grad}, x.grad_fn: {x.grad_fn}")

            # Third layer
            x = self.fc3(x)
            print(f"[DEBUG][PPOPolicy.forward][after fc3] x.requires_grad: {x.requires_grad}, x.grad_fn: {x.grad_fn}")
            x = self.ln3(x)
            print(f"[DEBUG][PPOPolicy.forward][after ln3] x.requires_grad: {x.requires_grad}, x.grad_fn: {x.grad_fn}")
            x = torch.relu(x)
            print(f"[DEBUG][PPOPolicy.forward][after relu3] x.requires_grad: {x.requires_grad}, x.grad_fn: {x.grad_fn}")
            x = self.dropout(x)
            print(f"[DEBUG][PPOPolicy.forward][after dropout3] x.requires_grad: {x.requires_grad}, x.grad_fn: {x.grad_fn}")
            
            # Policy and value heads
            logits = self.policy_head(x)
            if action_mask is not None:
                # Ensure mask is boolean
                action_mask = action_mask.bool()
                if len(action_mask.shape) == 1:
                    action_mask = action_mask.unsqueeze(0)
                # Check for shape mismatch before applying mask
                if logits.shape != action_mask.shape:
                    print(f"[WARNING] Logits shape {logits.shape} and mask shape {action_mask.shape} mismatch. Skipping mask.")
                else:
                    logits = logits.masked_fill(action_mask == 0, -1e9)
            probs = torch.softmax(logits, dim=-1)
            value = self.value_head(x)

            # Debug: print grad info for value
            print(f"[DEBUG][PPOPolicy.forward] value.requires_grad: {value.requires_grad}, value.grad_fn: {value.grad_fn}")
            return probs, value

def ppo_evrp(env, state_dim, action_dim, num_episodes=1000, gamma=0.99, clip_epsilon=0.2, lr=1e-3, update_epochs=4, batch_size=32, max_steps=200):
    import csv
    log_file = 'ppo_training_log.csv'
    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'reward', 'distance', 'customers_visited', 'penalties', 'loss', 'value_loss', 'policy_loss'])
    
    # Improved hyperparameters for better training stability
    lr = 3e-4  # Slightly higher learning rate
    clip_epsilon = 0.1  # Smaller clip epsilon for more stable updates
    gamma = 0.995  # Higher gamma for better long-term planning
    entropy_coef = 0.01  # Entropy coefficient for exploration
    batch_size_episodes = 32  # Larger batch size for more stable updates
    update_epochs = 3  # Fewer update epochs to prevent overfitting
    
    # Learning rate scheduler parameters
    lr_scheduler_patience = 50
    lr_scheduler_factor = 0.8
    min_lr = 1e-6
    
    # Early stopping parameters
    early_stopping_patience = 100
    early_stopping_min_delta = 0.01
    
    policy = PPOPolicy(state_dim, action_dim, dropout_rate=0.1)
    optimizer = optim.Adam(policy.parameters(), lr=lr, weight_decay=1e-4)  # Added weight decay
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=lr_scheduler_factor, 
        patience=lr_scheduler_patience, min_lr=min_lr
    )
    
    all_rewards = []
    all_distances = []
    all_customers_visited = []
    all_penalties = []
    all_losses = []
    batch_trajectories = []
    batch_rewards = []
    best_reward = -float('inf')
    best_policy_path = 'trained_policy_online.pth'
    
    # Load previous best policy if exists
    if os.path.exists(best_policy_path):
        try:
            policy.load_state_dict(torch.load(best_policy_path))
            print(f"[INFO] Loaded previous best policy from {best_policy_path} for continual learning.")
        except Exception as e:
            print(f"[WARN] Could not load previous policy: {e}")
    
    depot_idx = 0
    try:
        positions = env.positions if hasattr(env, 'positions') else None
        if positions is not None:
            depot = positions[depot_idx]
            max_dist = max(np.linalg.norm(depot - positions[i]) for i in range(len(positions)))
            norm_factor = max_dist * (action_dim-1) * 2
        else:
            norm_factor = 1000.0
    except Exception:
        norm_factor = 1000.0
    
    user_max_vehicles = None
    if hasattr(env, 'max_vehicles') and env.max_vehicles is not None:
        user_max_vehicles = env.max_vehicles
    else:
        try:
            from inspect import currentframe
            frame = currentframe()
            if 'max_vehicles' in frame.f_back.f_globals:
                user_max_vehicles = frame.f_back.f_globals['max_vehicles']
        except Exception:
            pass
    if user_max_vehicles is None:
        user_max_vehicles = 1
    
    # Curriculum learning with adaptive difficulty
    curriculum = True
    curriculum_episodes = 200  # Episodes for curriculum learning
    
    # Early stopping variables
    no_improvement_count = 0
    best_avg_reward = -float('inf')
    
    for episode in range(num_episodes):
        # Curriculum learning: gradually increase problem difficulty
        if hasattr(env, 'positions') and hasattr(env, 'demands') and hasattr(env, 'charging_stations'):
            if curriculum and episode < curriculum_episodes:
                # Gradually increase the number of customers
                progress = episode / curriculum_episodes
                max_customers = min(env.num_nodes-1, 3 + int(progress * (env.num_nodes-4)))
                perm = np.random.permutation(range(1, env.num_nodes))[:max_customers]
                perm = np.insert(perm, 0, 0)
                env.positions = env.positions[perm]
                env.demands = env.demands[perm]
                env.charging_stations = [i for i, idx in enumerate(perm) if idx in env.charging_stations]
            else:
                # Full problem after curriculum
                perm = np.arange(env.num_nodes)
                np.random.shuffle(perm[1:])
                env.positions = env.positions[perm]
                env.demands = env.demands[perm]
                env.charging_stations = [i for i, idx in enumerate(perm) if idx in env.charging_stations]
        
        num_nodes = action_dim
        demand_array = getattr(env, 'demands', np.zeros(num_nodes))
        charging_stations = getattr(env, 'charging_stations', [])
        customers_to_serve = set(i for i in range(1, num_nodes) if (demand_array[i] > 0 if i < len(demand_array) else False) and i not in charging_stations)
        served_customers = set()
        episode_reward = 0
        episode_log_probs = []
        episode_values = []
        episode_rewards = []
        episode_states = []
        episode_actions = []
        episode_masks = []
        
        # Improved reward shaping
        illegal_penalty = -0.5
        demand_reward = 0.1
        completion_bonus = 1.0
        visit_customer_reward = 0.05
        step_penalty = -0.01
        detour_penalty = -0.02
        distance_weight = -1.0 / norm_factor
        
        vehicle_count = 0
        episode_distance = 0.0
        episode_penalties = 0.0
        episode_customers_visited = set()
        
        while customers_to_serve and vehicle_count < user_max_vehicles:
            _reset_result = env.reset()
            num_nodes = action_dim
            current_node = env.current_node if hasattr(env, 'current_node') else 0
            visited_mask = np.array(env.visited) if hasattr(env, 'visited') else np.zeros(num_nodes)
            battery = env.battery if hasattr(env, 'battery') else 0.0
            if hasattr(battery, 'magnitude'):
                battery = battery.magnitude
            else:
                battery = float(battery)
            current_node_one_hot = np.zeros(num_nodes)
            current_node_one_hot[current_node] = 1.0
            state = np.concatenate([current_node_one_hot, [battery], visited_mask])
            if state.shape[0] != state_dim:
                raise ValueError(f"State shape mismatch: got {state.shape}, expected ({state_dim},)")
            done = False
            steps = 0
            illegal_move = False
            route_completed = False
            prev_node = current_node
            
            while not done and steps < max_steps:
                mask = env.get_action_mask()
                mask = np.asarray(mask).flatten()
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                mask_tensor = torch.FloatTensor(mask).unsqueeze(0)
                probs, value = policy(state_tensor, action_mask=mask_tensor)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().item()
                log_prob = dist.log_prob(torch.tensor(action))
                next_state_raw = env.step(action)
                if isinstance(next_state_raw, tuple):
                    next_state, reward, done, info = next_state_raw
                else:
                    next_state = next_state_raw
                    reward, done, info = 0, False, {}
                
                shaped_reward = 0.0
                if hasattr(env, 'positions') and prev_node is not None and action < len(env.positions):
                    dist = np.linalg.norm(env.positions[prev_node] - env.positions[action])
                    episode_distance += dist
                    shaped_reward += distance_weight * dist
                if hasattr(env, 'demands') and action < len(env.demands):
                    try:
                        demand_served = float(env.demands[action]) if env.demands[action] > 0 else 0.0
                    except Exception:
                        demand_served = 0.0
                    shaped_reward += demand_reward * demand_served
                if action in customers_to_serve and action not in episode_customers_visited:
                    shaped_reward += visit_customer_reward
                    episode_customers_visited.add(action)
                if isinstance(info, dict) and ('illegal' in info.get('status','') or 'infeasible' in info.get('status','')):
                    shaped_reward += illegal_penalty
                    episode_penalties += illegal_penalty
                    illegal_move = True
                if hasattr(env, 'visited') and action < len(env.visited) and env.visited[action]:
                    shaped_reward += detour_penalty
                    episode_penalties += detour_penalty
                shaped_reward += step_penalty
                if done and not illegal_move:
                    shaped_reward += completion_bonus
                    route_completed = True
                
                episode_rewards.append(shaped_reward)
                episode_log_probs.append(log_prob)
                episode_values.append(value)
                episode_states.append(state)
                episode_actions.append(action)
                episode_masks.append(mask)
                num_nodes = action_dim
                prev_node = action
                current_node = env.current_node if hasattr(env, 'current_node') else 0
                visited_mask = np.array(env.visited) if hasattr(env, 'visited') else np.zeros(num_nodes)
                battery = env.battery if hasattr(env, 'battery') else 0.0
                if hasattr(battery, 'magnitude'):
                    battery = battery.magnitude
                else:
                    battery = float(battery)
                current_node_one_hot = np.zeros(num_nodes)
                current_node_one_hot[current_node] = 1.0
                state = np.concatenate([current_node_one_hot, [battery], visited_mask])
                if state.shape[0] != state_dim:
                    raise ValueError(f"Next state shape mismatch: got {state.shape}, expected ({state_dim},)")
                episode_reward += shaped_reward
                steps += 1
                if action in customers_to_serve:
                    served_customers.add(action)
                    customers_to_serve.discard(action)
            vehicle_count += 1
        
        all_rewards.append(episode_reward)
        all_distances.append(episode_distance)
        all_customers_visited.append(len(episode_customers_visited))
        all_penalties.append(episode_penalties)
        
        batch_trajectories.append({
            'states': episode_states,
            'actions': episode_actions,
            'rewards': episode_rewards,
            'log_probs': episode_log_probs,
            'values': episode_values,
            'masks': episode_masks
        })
        batch_rewards.append(episode_reward)
        
        # Update policy with improved training loop
        if (episode+1) % batch_size_episodes == 0 or (episode+1) == num_episodes:
            if len(batch_trajectories) > 0:
                all_states = []
                all_actions = []
                all_rewards_flat = []
                all_log_probs = []
                all_values = []
                all_masks = []
                
                for traj in batch_trajectories:
                    all_states.extend(traj['states'])
                    all_actions.extend(traj['actions'])
                    all_rewards_flat.extend(traj['rewards'])
                    all_log_probs.extend(traj['log_probs'])
                    all_values.extend(traj['values'])
                    all_masks.extend(traj['masks'])
                
                if len(all_states) > 0 and len(all_values) > 0:
                    returns = []
                    G = 0
                    for r in reversed(all_rewards_flat):
                        G = r + gamma * G
                        returns.insert(0, G)
                    
                    returns = torch.tensor(returns, dtype=torch.float32)
                    values = torch.cat(all_values).squeeze(-1)
                    log_probs = torch.stack(all_log_probs)
                    states_tensor = torch.FloatTensor(all_states)
                    actions_tensor = torch.LongTensor(all_actions)
                    masks_tensor = torch.FloatTensor(all_masks)
                    advantages = returns - values.detach()
                    
                    # Normalize advantages for better training stability
                    if len(advantages) > 1:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                    
                    total_policy_loss = 0
                    total_value_loss = 0
                    
                    for _ in range(update_epochs):
                        probs, value = policy(states_tensor, action_mask=masks_tensor)
                        dist = torch.distributions.Categorical(probs)
                        new_log_probs = dist.log_prob(actions_tensor)
                        
                        # Calculate entropy for exploration
                        entropy = dist.entropy().mean()
                        
                        ratio = (new_log_probs - log_probs.detach()).exp()
                        surr1 = ratio * advantages
                        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
                        
                        policy_loss = -torch.min(surr1, surr2).mean() - entropy_coef * entropy
                        value_loss = (returns - value.squeeze(-1)).pow(2).mean()
                        
                        total_loss = policy_loss + 0.5 * value_loss
                        
                        optimizer.zero_grad()
                        total_loss.backward()
                        
                        # Gradient clipping for stability
                        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
                        
                        optimizer.step()
                        
                        total_policy_loss += policy_loss.item()
                        total_value_loss += value_loss.item()
                    
                    avg_policy_loss = total_policy_loss / update_epochs
                    avg_value_loss = total_value_loss / update_epochs
                    all_losses.append(avg_policy_loss + avg_value_loss)
                    
                    # Log detailed metrics
                    with open(log_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([episode+1, episode_reward, episode_distance, len(episode_customers_visited), episode_penalties, avg_policy_loss + avg_value_loss, avg_value_loss, avg_policy_loss])
                
                avg_reward = np.mean(batch_rewards) if batch_rewards else -float('inf')
                
                # Update learning rate scheduler
                scheduler.step(avg_reward)
                
                # Early stopping check
                if avg_reward > best_avg_reward + early_stopping_min_delta:
                    best_avg_reward = avg_reward
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    torch.save(policy.state_dict(), best_policy_path)
                    print(f"[INFO] New best policy saved with avg reward {avg_reward:.2f} at {best_policy_path}")
                
                # Early stopping
                if no_improvement_count >= early_stopping_patience:
                    print(f"[INFO] Early stopping triggered after {episode+1} episodes with no improvement for {early_stopping_patience} batches")
                    break
                
                batch_trajectories = []
                batch_rewards = []
        
        if (episode+1) % max(1, num_episodes//10) == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"[PPO] Episode {episode+1}/{num_episodes}, Reward: {episode_reward:.2f}, Distance: {episode_distance:.2f}, Customers Visited: {len(episode_customers_visited)}, Penalties: {episode_penalties:.2f}, LR: {current_lr:.2e}")
    
    # Plot learning curves
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot rewards
    axs[0,0].plot(all_rewards, label='Episode Reward', alpha=0.6)
    if len(all_rewards) > 10:
        # Plot moving average
        window = min(50, len(all_rewards) // 10)
        moving_avg = np.convolve(all_rewards, np.ones(window)/window, mode='valid')
        axs[0,0].plot(range(window-1, len(all_rewards)), moving_avg, label=f'Moving Avg ({window})', linewidth=2)
    axs[0,0].set_xlabel('Episode')
    axs[0,0].set_ylabel('Reward')
    axs[0,0].set_title('PPO Learning Curve')
    axs[0,0].legend()
    axs[0,0].grid(True)
    
    # Plot distances
    axs[0,1].plot(all_distances, label='Total Distance', color='orange', alpha=0.6)
    axs[0,1].set_xlabel('Episode')
    axs[0,1].set_ylabel('Distance')
    axs[0,1].set_title('Distance per Episode')
    axs[0,1].legend()
    axs[0,1].grid(True)
    
    # Plot customers visited
    axs[1,0].plot(all_customers_visited, label='Customers Visited', color='green', alpha=0.6)
    axs[1,0].set_xlabel('Episode')
    axs[1,0].set_ylabel('Customers')
    axs[1,0].set_title('Customers Visited per Episode')
    axs[1,0].legend()
    axs[1,0].grid(True)
    
    # Plot losses
    if all_losses:
        axs[1,1].plot(all_losses, label='Total Loss', color='red', alpha=0.6)
        axs[1,1].set_xlabel('Update Step')
        axs[1,1].set_ylabel('Loss')
        axs[1,1].set_title('Training Loss')
        axs[1,1].legend()
        axs[1,1].grid(True)
    
    plt.tight_layout()
    plt.savefig('ppo_learning_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return policy, all_rewards
import numpy as np
from scipy.spatial import distance_matrix
import random
from units import ureg

def create_tour(num_nodes):
    """
    Create a tour starting at node 0 (depot) and visiting all customer nodes randomly.

    :param int num_nodes: Total number of nodes (including depot)
    :return list: Tour starting with node 0 followed by a random permutation of customer nodes
    """
    customer_nodes = list(range(1, num_nodes))
    random.shuffle(customer_nodes)
    tour = [0] + customer_nodes
    print(f"Generated tour: {tour}")  # Debug statement
    return tour

def swap_2opt(tour, i, k):
    """
    Apply a 2-opt swap to the tour by reversing the segment between indices i and k.

    :param list tour: Current tour (starting with depot)
    :param int i: Start index of the segment to reverse
    :param int k: End index of the segment to reverse
    :return list: New tour after 2-opt swap
    """
    new_tour = tour[:i] + tour[i:k+1][::-1] + tour[k+1:]
    return new_tour

class EVEnvironment:
    def start_new_vehicle(self):
        """
        Reset only the vehicle state (battery, position, tour, time), but do NOT reset the visited mask.
        This is used for multi-vehicle rollouts: each new vehicle starts at depot, but only unvisited customers remain.
        """
        self.tour = [0]  # Start at depot
        self.soc = self.battery_capacity
        self.current_pos = 0
        self.time = 0.0 * ureg.hour
        print(f"[Multi-vehicle] New vehicle started: tour={self.tour}, current_pos={self.current_pos}, visited={self.visited}")
        return self.get_state()
    @property
    def battery(self):
        # For compatibility with RL code: always return float (watt_hour)
        return self.soc.magnitude if hasattr(self.soc, 'magnitude') else float(self.soc)

    @property
    def battery_capacity_value(self):
        # Return battery capacity as a float (watt_hour)
        return self.battery_capacity.magnitude if hasattr(self.battery_capacity, 'magnitude') else float(self.battery_capacity)

    @property
    def current_node(self):
        return self.current_pos

    @property
    def time_value(self):
        return self.time.magnitude if hasattr(self.time, 'magnitude') else float(self.time)

    @property
    def visited_nodes(self):
        return self.visited
    def get_action_mask(self):
        """
        Returns a numpy array of shape (num_nodes,) with 1 for valid actions and 0 for invalid.
        Blocks already visited customers, and depot (0) unless all customers are visited.
        """
        num_nodes = len(self.visited)
        mask = np.ones(num_nodes, dtype=np.float32)
        # Block depot (node 0) unless all customers are visited
        if not all(self.visited[1:]):
            mask[0] = 0.0
        # Block already visited customers
        for i in range(1, num_nodes):
            if self.visited[i]:
                mask[i] = 0.0
        return mask
    def __init__(self, positions, charging_stations, battery_capacity=50, consumption_rate=0.2, min_soc=0.1, charging_time=0.5, max_vehicles=None):
        self.positions = positions
        print(f"Positions shape: {self.positions.shape}")  # Debug statement
        self.num_nodes = len(positions)
        self.num_customers = self.num_nodes - len(charging_stations) if charging_stations else self.num_nodes - 1
        self.charging_stations = charging_stations
        self.battery_capacity = battery_capacity * ureg.watt_hour if not hasattr(battery_capacity, 'units') else battery_capacity
        self.consume_rate = consumption_rate * ureg.watt_hour / ureg.kilometer if not hasattr(consumption_rate, 'units') else consumption_rate
        self.min_soc = min_soc * self.battery_capacity
        self.charging_time = charging_time * ureg.hour if not hasattr(charging_time, 'units') else charging_time
        self.distances = np.linalg.norm(self.positions[:, np.newaxis] - self.positions, axis=2) * ureg.kilometer
        self.max_vehicles = max_vehicles  # Store max_vehicles for RL agent use
        self.tour = [0]  # Initialize tour starting at depot
        self.soc = self.battery_capacity
        self.current_pos = 0
        self.visited = [False] * self.num_nodes
        self.visited[0] = True
        self.time = 0.0 * ureg.hour

    def reset(self):
        """
        Reset the environment to start a new episode (all customers unvisited).
        """
        self.visited = [False] * self.num_nodes
        self.visited[0] = True
        return self.start_new_vehicle()

    def step(self, next_node):
        """
        Take a step in the environment by moving to the next node.

        :param int next_node: The node to visit next
        :return tuple: (state, reward, done, info)
        """
        print(f"Attempting to move to node {next_node} from {self.current_pos}, SOC={self.soc}")
        # Allow returning to depot if all customers are visited
        if next_node == 0 and all(self.visited[1:self.num_customers + 1]):
            distance = self.distances[self.current_pos][next_node]
            consumption = distance * self.consume_rate
            if self.soc - consumption < self.min_soc:
                print(f"Warning: SOC {self.soc} too low to return to depot (needs {consumption})")
                return self.get_state(), 0, False, {"error": "Cannot return to depot: Battery too low"}
            self.tour.append(next_node)
            self.soc -= consumption
            self.time += (distance / (30 * ureg.kilometer / ureg.hour))  # Assume 30 km/h speed for time
            self.current_pos = next_node
            print(f"Tour complete: {self.tour}, Total time={self.time}")
            reward = -self.time.magnitude * 100
            return self.get_state(), reward, True, {"time": self.time}

        if self.visited[next_node]:
            print(f"Error: Node {next_node} already visited")
            return self.get_state(), 0, False, {"error": "Node already visited"}

        if next_node == 0:
            if not all(self.visited[1:self.num_customers + 1]):
                print(f"Error: Attempted to return to depot before visiting all customers")
                return self.get_state(), 0, False, {"error": "Cannot return to depot before visiting all customers"}

        distance = self.distances[self.current_pos][next_node]
        consumption = distance * self.consume_rate
        print(f"Distance to {next_node}: {distance}, Consumption: {consumption}")

        if self.soc - consumption < self.min_soc:
            print(f"Warning: SOC {self.soc} too low to reach node {next_node} (needs {consumption})")
            if self.charging_stations:
                nearest_charge = min(
                    [(cs, self.distances[self.current_pos][cs]) for cs in self.charging_stations if not self.visited[cs]],
                    key=lambda x: x[1],
                    default=None
                )
                if nearest_charge:
                    cs_idx, cs_distance = nearest_charge
                    cs_consumption = cs_distance * self.consume_rate
                    if self.soc - cs_consumption >= self.min_soc:
                        print(f"Routing to charging station {cs_idx}, distance={cs_distance}")
                        self.tour.append(cs_idx)
                        self.soc -= cs_consumption
                        self.time += (cs_distance / (30 * ureg.kilometer / ureg.hour)) + self.charging_time
                        self.visited[cs_idx] = True
                        self.current_pos = cs_idx
                        self.soc = self.battery_capacity
                        return self.get_state(), -self.time.magnitude * 100, False, {"current_pos": self.current_pos}
                    else:
                        print(f"Error: Cannot reach charging station {cs_idx}, SOC too low")
                else:
                    print(f"Error: No available charging stations")
                return self.get_state(), 0, False, {"error": "Cannot travel to node: Battery too low"}

        self.tour.append(next_node)
        self.soc -= consumption
        self.time += distance / (30 * ureg.kilometer / ureg.hour)
        self.visited[next_node] = True
        self.current_pos = next_node
        print(f"Updated tour: {self.tour}, SOC={self.soc}, Time={self.time}")

        return self.get_state(), -self.time.magnitude * 100, False, {"current_pos": self.current_pos}

    def get_state(self):
        """
        Get the current state of the environment.

        :return tuple: (tour, soc, current_pos, visited)
        """
        return self.tour, self.soc, self.current_pos, self.visited