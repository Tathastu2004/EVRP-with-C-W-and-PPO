


# === Imports and Scenario Setup ===
import numpy as np
import os
import torch
import csv
import matplotlib.pyplot as plt
from mp_utils import clarke_wright_evrp, EVEnvironment, ppo_evrp
try:
    from mp_utils import debug_print
except ImportError:
    def debug_print(*args, **kwargs):
        print(*args, **kwargs)

# Import the GIF functionality
try:
    from make_gif import make_gif_from_tour, make_gif_from_tour_advanced
except ImportError:
    # Fallback if make_gif.py is not available
    def make_gif_from_tour(positions, tour_sequence, gif_path):
        """
        Fallback function that just saves tour data to a text file.
        """
        txt_path = gif_path.replace('.gif', '_tour.txt')
        try:
            with open(txt_path, 'w') as f:
                f.write(f"Tour sequence for {gif_path}\n")
                f.write("=" * 50 + "\n")
                for i, tour in enumerate(tour_sequence):
                    f.write(f"Step {i+1}: {tour}\n")
            print(f"[INFO] Tour sequence saved to {txt_path} (GIF generation not available)")
        except Exception as e:
            print(f"[WARNING] Could not save tour sequence: {e}")
    
    def make_gif_from_tour_advanced(positions, tour_sequence, gif_path, **kwargs):
        return make_gif_from_tour(positions, tour_sequence, gif_path)

try:
    from units import ureg
except ImportError:
    # fallback if units.py is missing
    class DummyUreg:
        kilometer = 1.0
    ureg = DummyUreg()


# === Load scenario from test_case_evrp.tsp ===
def load_evrp_tsp(filename):
    positions = []
    demands = []
    charging_stations = []
    vehicle_capacity = None
    battery_capacity = None
    consumption_rate = None
    min_soc = 0.1
    charging_time = 1.0
    max_vehicles = 2
    num_customers = 0
    with open(filename, 'r') as f:
        lines = f.readlines()
    node_section = False
    demand_section = False
    for line in lines:
        line = line.strip()
        if line.startswith('CAPACITY'):
            vehicle_capacity = float(line.split()[-1])
        elif line.startswith('BATTERY_CAPACITY'):
            battery_capacity = float(line.split()[-1])
        elif line.startswith('CONSUMPTION_RATE'):
            consumption_rate = float(line.split()[-1])
        elif line.startswith('MIN_SOC'):
            min_soc = float(line.split()[-1])
        elif line.startswith('CHARGING_TIME'):
            charging_time = float(line.split()[-1])
        elif line.startswith('MAX_VEHICLES'):
            max_vehicles = int(line.split()[-1])
        elif line.startswith('NODE_COORD_SECTION'):
            node_section = True
            continue
        elif line.startswith('DEMAND_SECTION'):
            node_section = False
            demand_section = True
            continue
        elif line.startswith('CHARGING_STATIONS'):
            demand_section = False
            cs_indices = line.split(':')[-1].strip()
            if cs_indices:
                charging_stations = [int(idx) for idx in cs_indices.split()]
            continue
        elif line.startswith('EOF'):
            break
        if node_section and line:
            parts = line.split()
            if len(parts) >= 3:
                positions.append([float(parts[1]), float(parts[2])])
        if demand_section and line:
            parts = line.split()
            if len(parts) >= 2:
                demands.append(float(parts[1]))
    positions = np.array(positions)
    demands = np.array(demands)
    num_customers = len(demands) - 1
    positive_demand_customers = [i for i in range(1, len(demands)) if demands[i] > 0]
    return (num_customers, positions, demands, charging_stations, vehicle_capacity, battery_capacity, consumption_rate, min_soc, charging_time, max_vehicles, positive_demand_customers)


# Load scenario
(num_customers, original_positions, demands, charging_stations, vehicle_capacity, battery_capacity, consumption_rate, min_soc, charging_time, max_vehicles, positive_demand_customers) = load_evrp_tsp('test_case_evrp.tsp')



# Always prompt user for all scenario constraints, regardless of TSP file
def prompt_float(prompt, default):
    while True:
        val = input(f"{prompt} [{default}]: ").strip()
        if val == '':
            return float(default)
        try:
            return float(val)
        except Exception:
            print("Invalid input. Please enter a number.")

def prompt_int(prompt, default):
    while True:
        val = input(f"{prompt} [{default}]: ").strip()
        if val == '':
            return int(default)
        try:
            return int(val)
        except Exception:
            print("Invalid input. Please enter an integer.")

print("[INPUT] Please enter the following scenario constraints (press Enter to use the default shown in brackets):")
battery_capacity = prompt_float("Battery capacity (watt_hour)", battery_capacity or 6000.0)
vehicle_capacity = prompt_float("Vehicle load capacity (kilogram)", vehicle_capacity or 100.0)
consumption_rate = prompt_float("Consumption rate (watt_hour / kilometer)", consumption_rate or 1.0)
min_soc = prompt_float("Minimum state of charge (fraction, e.g., 0.1)", min_soc or 0.1)
charging_time = prompt_float("Charging time (hour)", charging_time or 0.5)
max_vehicles = prompt_int("Maximum number of vehicles", max_vehicles or 10)

print("\nYou have entered the following constraints:")
print(f"  Battery capacity: {battery_capacity} watt_hour")
print(f"  Vehicle load capacity: {vehicle_capacity} kilogram")
print(f"  Consumption rate: {consumption_rate} watt_hour / kilometer")
print(f"  Minimum SoC: {min_soc} (fraction)")
print(f"  Charging time: {charging_time} hour")
print(f"  Maximum number of vehicles: {max_vehicles}")

# --- End scenario setup ---
print(f"[INFO] Current working directory: {os.getcwd()}")

# === Mode Selection ===
mode = input("Enter mode ('cw' for Clarke & Wright, 'ppo' for Proximal Policy Optimization): ").strip().lower()

if mode == "cw":
    if clarke_wright_evrp is None:
        print("[ERROR] Clarke & Wright's Savings Algorithm is not available in mp_utils.py. Please implement or import clarke_wright_evrp.")
        import sys
        sys.exit(1)
    # --- Clarke & Wright's Savings Algorithm for EVRP ---
    debug_print("\n=== Clarke & Wright's Savings Algorithm for EVRP ===")
    cw_routes, strict_vehicle_limit = clarke_wright_evrp(
        original_positions, demands, 0, charging_stations,
        vehicle_capacity, battery_capacity, consumption_rate, min_soc, charging_time, max_vehicles=max_vehicles
    )
    
    # --- Calculate C&W Statistics ---
    total_vehicles_cw = len(cw_routes)
    total_demand_served_cw = 0.0
    total_time_cw = 0.0
    total_battery_consumed_cw = 0.0
    
    # Calculate statistics for each route
    for route in cw_routes:
        route_demand = 0.0
        route_distance = 0.0
        route_time = 0.0
        route_battery = 0.0
        
        # Calculate route statistics
        for i in range(len(route) - 1):
            from_node = route[i]
            to_node = route[i + 1]
            
            # Calculate distance
            distance = np.linalg.norm(original_positions[from_node] - original_positions[to_node])
            route_distance += distance
            
            # Calculate time (assuming 1 hour per unit distance)
            route_time += distance / 30.0  # Assuming 30 km/h average speed
            
            # Calculate battery consumption
            battery_consumed = distance * consumption_rate
            route_battery += battery_consumed
            
            # Add demand if it's a customer node (not depot or charging station)
            if to_node != 0 and to_node not in charging_stations and to_node < len(demands):
                route_demand += demands[to_node] if hasattr(demands[to_node], 'magnitude') else demands[to_node]
        
        total_demand_served_cw += route_demand
        total_time_cw += route_time
        total_battery_consumed_cw += route_battery
    
    # --- Save C&W Statistics to File ---
    cw_stats_file = os.path.join("data", "cw_solution_stats.txt")
    try:
        with open(cw_stats_file, 'w') as f:
            f.write("=== Clarke & Wright Savings Algorithm - Solution Statistics ===\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total Vehicles Used: {total_vehicles_cw}\n")
            f.write(f"Total Demand Served: {total_demand_served_cw:.2f} kilogram\n")
            f.write(f"Total Time Taken: {total_time_cw:.2f} hour\n")
            f.write(f"Total Battery Consumed: {total_battery_consumed_cw:.2f} watt_hour\n\n")
            
            f.write("=== Route Details ===\n")
            f.write("-" * 30 + "\n")
            for i, route in enumerate(cw_routes, 1):
                route_demand = sum(demands[node] if node < len(demands) and node != 0 and node not in charging_stations else 0 for node in route)
                route_distance = sum(np.linalg.norm(original_positions[route[j]] - original_positions[route[j+1]]) for j in range(len(route)-1))
                route_time = route_distance / 30.0  # Assuming 30 km/h average speed
                route_battery = route_distance * consumption_rate
                
                f.write(f"Route {i}: {route}\n")
                f.write(f"  - Demand Served: {route_demand:.2f} kilogram\n")
                f.write(f"  - Distance: {route_distance:.2f} kilometer\n")
                f.write(f"  - Time: {route_time:.2f} hour\n")
                f.write(f"  - Battery Consumed: {route_battery:.2f} watt_hour\n\n")
        
        print(f"[STATS] C&W solution statistics saved to {cw_stats_file}")
    except Exception as e:
        print(f"[ERROR] Could not save C&W statistics: {e}")
    
    # --- GIF: Create and save GIF for C&W ---
    cw_tour_sequence = []
    for res in cw_routes:
        # Add each route step-by-step for GIF (showing progress)
        for i in range(2, len(res)+1):
            cw_tour_sequence.append(res[:i])
    gif_path_cw = os.path.join("data", "cw_solution.gif")
    try:
        make_gif_from_tour_advanced(original_positions, cw_tour_sequence, gif_path_cw)
        print(f"[GIF] C&W tour animation saved to {gif_path_cw}")
    except Exception as e:
        print(f"[GIF] Error creating C&W GIF: {e}")
    
    # Print summary
    print(f"\n=== C&W Solution Summary ===")
    print(f"Total Vehicles: {total_vehicles_cw}")
    print(f"Total Demand Served: {total_demand_served_cw:.2f} kilogram")
    print(f"Total Time: {total_time_cw:.2f} hour")
    print(f"Total Battery Consumed: {total_battery_consumed_cw:.2f} watt_hour")

elif mode == "ppo":
    print("Running PPO-based EVRP...")
    ppo_positions = original_positions
    ppo_demands = demands
    ppo_charging = charging_stations
    env = EVEnvironment(ppo_positions, ppo_charging, battery_capacity=battery_capacity, consumption_rate=consumption_rate, min_soc=min_soc, charging_time=charging_time, max_vehicles=max_vehicles)
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        state = reset_result[0]
    else:
        state = reset_result
    num_nodes = len(ppo_positions)
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
    state_dim = state.shape[0]
    action_dim = num_nodes
    # Pass max_vehicles explicitly to PPO agent via env
    # max_vehicles is now always set in env and used in PPO
    # PPO with online/continual learning: always loads previous best policy and saves new best
    policy, rewards = ppo_evrp(env, state_dim, action_dim, num_episodes=1000, gamma=0.99, lr=1e-3, max_steps=4*num_nodes+20)
    # RL evaluation/rollout: use up to max_vehicles, just like PPO training
    env_eval = EVEnvironment(ppo_positions, ppo_charging, battery_capacity=battery_capacity, consumption_rate=consumption_rate, min_soc=min_soc, charging_time=charging_time, max_vehicles=max_vehicles)
    num_nodes_eval = len(ppo_positions)
    demand_array = ppo_demands if hasattr(ppo_demands, 'shape') else np.array(ppo_demands)
    customers_to_serve = set(i for i in range(1, num_nodes_eval) if demand_array[i] > 0 and i not in ppo_charging)
    served_customers = set()
    total_demand_served = 0
    total_battery_consumed = 0.0
    total_time = 0.0
    charging_events = 0
    total_reward = 0
    steps = 0
    # --- GIF: Collect PPO rollout steps ---
    ppo_tour_sequence = []
    route_segments = []
    vehicle_count = 0
    max_steps_eval = 2 * (num_nodes_eval - 1) + 10
    while customers_to_serve and vehicle_count < max_vehicles:
        # Only reset vehicle state, not visited mask
        reset_result = env_eval.start_new_vehicle()
        if isinstance(reset_result, tuple):
            state = reset_result[0]
        else:
            state = reset_result
        current_node = env_eval.current_node if hasattr(env_eval, 'current_node') else 0
        visited_mask = np.array(env_eval.visited) if hasattr(env_eval, 'visited') else np.zeros(num_nodes_eval)
        battery = env_eval.battery if hasattr(env_eval, 'battery') else 0.0
        if hasattr(battery, 'magnitude'):
            battery = battery.magnitude
        else:
            battery = float(battery)
        current_node_one_hot = np.zeros(num_nodes_eval)
        current_node_one_hot[current_node] = 1.0
        state = np.concatenate([current_node_one_hot, [battery], visited_mask])
        curr_route = [current_node]
        curr_demand = 0
        curr_battery_used = 0.0
        curr_time = 0.0
        prev_battery = battery
        prev_time = getattr(env_eval, 'time', 0.0)
        done = False
        local_steps = 0
        # --- GIF: Add initial node for this vehicle ---
        ppo_tour_sequence.append([current_node])
        while not done and local_steps < max_steps_eval:
            action_mask = np.ones(num_nodes_eval, dtype=np.float32)
            if not all(env_eval.visited[1:]):
                action_mask[0] = 0.0
            for i in range(1, num_nodes_eval):
                if env_eval.visited[i]:
                    action_mask[i] = 0.0
            # --- Check if any unvisited customer is reachable ---
            feasible_customers = [i for i in range(1, num_nodes_eval) if not env_eval.visited[i] and action_mask[i] > 0]
            can_reach_any = False
            for cust in feasible_customers:
                # Estimate if reachable: check battery constraint
                try:
                    dist = env_eval.distances[env_eval.current_node][cust]
                    consumption = dist * env_eval.consume_rate
                    soc = env_eval.soc if hasattr(env_eval, 'soc') else env_eval.battery
                    min_soc = env_eval.min_soc if hasattr(env_eval, 'min_soc') else 0.0
                    if soc - consumption >= min_soc:
                        can_reach_any = True
                        break
                except Exception:
                    continue
            if not can_reach_any:
                # No more reachable customers, force depot return and break
                if curr_route[-1] != 0:
                    next_state, reward, done, info = env_eval.step(0)
                    curr_time += getattr(env_eval, 'time', prev_time) - prev_time
                    curr_route.append(0)
                    ppo_tour_sequence.append(curr_route.copy())
                break
            action_mask_tensor = torch.tensor(action_mask, dtype=torch.float32).unsqueeze(0)
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            probs, value = policy(state_tensor, action_mask=action_mask_tensor)
            action = torch.argmax(probs, dim=-1).item()
            next_state, reward, done, info = env_eval.step(action)
            shaped_reward = reward
            if action == 0 and not all(env_eval.visited[1:]):
                shaped_reward -= 100
            if action == 0 and local_steps > 0 and sum(env_eval.visited[1:]) == 0:
                shaped_reward -= 100
            if action != 0 and (demand_array[action] > 0):
                shaped_reward += 50
            shaped_reward -= 1
            if isinstance(info, dict) and ('illegal' in info.get('status','') or 'infeasible' in info.get('status','')):
                shaped_reward -= 500
            total_reward += shaped_reward
            if action != 0 and (demand_array[action] > 0 if action < len(demand_array) else False) and action not in served_customers:
                curr_demand += demand_array[action]
                served_customers.add(action)
                customers_to_serve.discard(action)
            try:
                current_battery = env_eval.battery if hasattr(env_eval, 'battery') else prev_battery
                if current_battery < prev_battery:
                    curr_battery_used += float(prev_battery - current_battery)
                if current_battery > prev_battery:
                    charging_events += 1
                prev_battery = current_battery
            except Exception:
                pass
            try:
                current_time = getattr(env_eval, 'time', prev_time)
            except Exception:
                current_time = prev_time
            if action == 0 and len(curr_route) > 1:
                curr_time += current_time - prev_time
                break
            else:
                curr_time += current_time - prev_time
                curr_route.append(action)
                # --- GIF: Add current step ---
                ppo_tour_sequence.append(curr_route.copy())
            prev_time = current_time
            current_node = env_eval.current_node if hasattr(env_eval, 'current_node') else action
            visited_mask = np.array(env_eval.visited) if hasattr(env_eval, 'visited') else np.zeros(num_nodes_eval)
            battery = env_eval.battery if hasattr(env_eval, 'battery') else 0.0
            if hasattr(battery, 'magnitude'):
                battery = battery.magnitude
            else:
                battery = float(battery)
            current_node_one_hot = np.zeros(num_nodes_eval)
            current_node_one_hot[current_node] = 1.0
            state = np.concatenate([current_node_one_hot, [battery], visited_mask])
            local_steps += 1
            steps += 1
        if len(curr_route) > 1:
            route_segments.append({
                'route': curr_route + [0] if curr_route[-1] != 0 else curr_route,
                'demand_served': curr_demand,
                'battery_used': curr_battery_used,
                'time': curr_time
            })
            total_demand_served += curr_demand
            total_battery_consumed += curr_battery_used
            total_time += curr_time
        vehicle_count += 1
    # --- GIF: Create and save GIF for PPO ---
    gif_path_ppo = os.path.join("data", "ppo_solution.gif")
    try:
        make_gif_from_tour_advanced(ppo_positions, ppo_tour_sequence, gif_path_ppo)
        print(f"[GIF] PPO tour animation saved to {gif_path_ppo}")
    except Exception as e:
        print(f"[GIF] Error creating PPO GIF: {e}")
    
    # --- Save PPO Statistics to File ---
    ppo_stats_file = os.path.join("data", "ppo_solution_stats.txt")
    try:
        with open(ppo_stats_file, 'w') as f:
            f.write("=== PPO (Proximal Policy Optimization) - Solution Statistics ===\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total Vehicles Used: {len(route_segments)}\n")
            f.write(f"Total Demand Served: {total_demand_served:.2f} kilogram\n")
            f.write(f"Total Time Taken: {total_time:.2f} hour\n")
            f.write(f"Total Battery Consumed: {total_battery_consumed:.2f} watt_hour\n")
            f.write(f"Total Reward: {total_reward:.2f}\n")
            f.write(f"Steps Taken: {steps}\n")
            f.write(f"Charging Events: {charging_events}\n\n")
            
            f.write("=== Route Details ===\n")
            f.write("-" * 30 + "\n")
            for idx, seg in enumerate(route_segments, 1):
                f.write(f"Route {idx}: {seg['route']}\n")
                f.write(f"  - Demand Served: {seg['demand_served']:.2f} kilogram\n")
                f.write(f"  - Time: {seg['time']:.2f} hour\n")
                f.write(f"  - Battery Consumed: {seg['battery_used']:.2f} watt_hour\n\n")
        
        print(f"[STATS] PPO solution statistics saved to {ppo_stats_file}")
    except Exception as e:
        print(f"[ERROR] Could not save PPO statistics: {e}")
    
    # Save detailed results to the existing file
    with open("ppo_results.txt", "w") as f:
        f.write("\n=== PPO Agent Online Run (Detailed) ===\n")
        for idx, seg in enumerate(route_segments, 1):
            f.write(f"PPO Route {idx}: {seg['route']}, Demand Served: {seg['demand_served']} kilogram, Time: {seg['time']} hour, Battery Used: {seg['battery_used']} watt_hour\n")
        f.write(f"Total Vehicles (PPO): {len(route_segments)}\n")
        f.write(f"Total Demand Served (PPO): {total_demand_served} kilogram\n")
        f.write(f"Total Time (PPO): {total_time} hour\n")
        f.write(f"Total Battery Consumed (PPO): {total_battery_consumed} watt_hour\n")
        f.write(f"Total Reward: {total_reward}\n")
        f.write(f"Steps Taken: {steps}\n")
        f.write(f"Charging Events: {charging_events}\n")
    
    # Print summary
    print(f"\n=== PPO Solution Summary ===")
    print(f"Total Vehicles: {len(route_segments)}")
    print(f"Total Demand Served: {total_demand_served:.2f} kilogram")
    print(f"Total Time: {total_time:.2f} hour")
    print(f"Total Battery Consumed: {total_battery_consumed:.2f} watt_hour")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Steps Taken: {steps}")
    print(f"Charging Events: {charging_events}")
else:
    print("Invalid mode selected.")

# --- Scenario Diagnosis and Recommendations ---
# Only run this section if cw_routes is defined (i.e., in 'cw' mode)
if 'cw_routes' in locals():
    # Find all customers with positive demand
    all_customers = set(i for i in range(len(demands)) if i != 0 and i not in charging_stations and demands[i] > 0)
    # Find all customers served in the routes
    served_customers = set()
    for route in cw_routes:
        for node in route:
            if node in all_customers:
                served_customers.add(node)
    unserved_customers = all_customers - served_customers

    # Write diagnosis and recommendations to the results file
    def _scalar(x):
        return x.magnitude if hasattr(x, 'magnitude') else x
    with open("evrp_charging_results.txt", "a") as f:
        f.write("\n=== Scenario Diagnosis and Recommendations ===\n")
        batt_cap = _scalar(battery_capacity)
        veh_cap = _scalar(vehicle_capacity)
        if not unserved_customers:
            f.write("All customers with positive demand were served. Your scenario and input are feasible!\n")
        else:
            for cust in sorted(unserved_customers):
                cust_pos = original_positions[cust]
                dist_to_cust = np.linalg.norm(original_positions[0] - cust_pos)  # kilometers (already scalar)
                # Minimum battery needed (out + back + reserve)
                min_batt_needed = 2 * dist_to_cust * consumption_rate + min_soc * batt_cap
                # Nearest charging station distance
                min_dist_cs = float('inf')
                closest_cs = None
                for cs in charging_stations:
                    d = np.linalg.norm(cust_pos - original_positions[cs])
                    if d < min_dist_cs:
                        min_dist_cs = d
                        closest_cs = cs
                demand_cust = _scalar(demands[cust]) if cust < len(demands) else 0
                if demand_cust > veh_cap:
                    f.write(f"- Customer {cust} (demand {demand_cust}) is unserved. Reason: Demand exceeds vehicle capacity ({veh_cap}).\n")
                    f.write(f"  Recommendation: Increase vehicle load capacity to at least {demand_cust}.\n")
                    continue
                usable_energy = batt_cap * (1 - min_soc)
                need_one_way = dist_to_cust * consumption_rate
                can_reach_from_depot = need_one_way <= usable_energy
                can_return_to_depot = need_one_way <= usable_energy  # symmetric assumption
                need_to_cs = min_dist_cs * consumption_rate if min_dist_cs < float('inf') else float('inf')
                can_reach_cs = need_to_cs <= usable_energy
                if not can_reach_from_depot:
                    f.write(f"- Customer {cust} (demand {demand_cust}) is unserved. Reason: Battery capacity ({batt_cap}) too low to reach (distance {dist_to_cust:.2f} km, need {need_one_way:.1f} > usable {usable_energy:.1f}).\n")
                    f.write(f"  Recommendation: Increase battery to >= {min_batt_needed:.1f} or add a charging station nearer (current nearest {closest_cs} at {min_dist_cs:.2f} km).\n")
                elif not can_reach_cs and not can_return_to_depot:
                    f.write(f"- Customer {cust} (demand {demand_cust}) is unserved. Reason: No reachable charging station (nearest {closest_cs} at {min_dist_cs:.2f} km).\n")
                    f.write(f"  Recommendation: Add a charging station at or near node {cust}.\n")
                else:
                    f.write(f"- Customer {cust} (demand {demand_cust}) is unserved. Reason: Route construction or vehicle/merge limits.\n")
                    f.write(f"  Recommendation: Increase max_vehicles or adjust scenario parameters (capacity, charging).\n")
        # Check if vehicle limit is a factor
        min_vehicles_needed = int(np.ceil(sum(_scalar(demands[i]) for i in all_customers) / veh_cap))
        if max_vehicles < min_vehicles_needed:
            f.write(f"- The maximum number of vehicles allowed ({max_vehicles}) is less than the minimum needed ({min_vehicles_needed}) to serve all demand.\n")
            f.write(f"  Recommendation: Increase max_vehicles to at least {min_vehicles_needed}.\n")
        elif max_vehicles > min_vehicles_needed:
            f.write(f"- The maximum number of vehicles allowed ({max_vehicles}) is more than required for this scenario.\n")
            f.write(f"  Note: The minimum number of vehicles needed to serve all demand is {min_vehicles_needed}. You may reduce max_vehicles to {min_vehicles_needed} for optimal resource use.\n")

# (The old greedy scenario logic is now removed/commented out)

# --- End of script ---

# --- Plotting Section ---
# --- Side-by-side CW vs RL Plot ---
import matplotlib.patches as mpatches
def plot_cw_vs_rl(cw_results=None, rl_results=None, positions=None, charging_stations=None, vehicle_capacity=None, battery_capacity=None, num_customers=None):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    # CW plot
    ax = axes[0]
    ax.set_title("Clarke & Wright's Savings Algorithm")
    ax.scatter(positions[1:num_customers+1, 0], positions[1:num_customers+1, 1], c='blue', label='Customers')
    ax.scatter(positions[0, 0], positions[0, 1], c='red', marker='^', label='Depot')
    if charging_stations:
        ax.scatter(positions[charging_stations, 0], positions[charging_stations, 1], c='green', marker='s', label='Charging Station')
    if cw_results:
        for res in cw_results:
            tour_points = positions[res]
            ax.plot(tour_points[:, 0], tour_points[:, 1], '-', label='Route', alpha=0.7)
    for i, (x, y) in enumerate(positions):
        ax.text(x, y + 0.5, f'{i}', fontsize=8, ha='center')
    ax.legend()
    ax.grid(True)
    # RL plot
    ax = axes[1]
    ax.set_title("RL Agent Solution")
    ax.scatter(positions[1:num_customers+1, 0], positions[1:num_customers+1, 1], c='blue', label='Customers')
    ax.scatter(positions[0, 0], positions[0, 1], c='red', marker='^', label='Depot')
    if charging_stations:
        ax.scatter(positions[charging_stations, 0], positions[charging_stations, 1], c='green', marker='s', label='Charging Station')
    # Parse RL routes from PPO results file (look for 'PPO Route')
    rl_routes = []
    try:
        with open("ppo_results.txt", "r") as f:
            for line in f:
                if line.startswith("PPO Route"):
                    route_str = line.split(":")[1].split(", Demand")[0].strip()
                    route = eval(route_str)
                    rl_routes.append(route)
    except Exception:
        pass
    for route in rl_routes:
        tour_points = positions[route]
        ax.plot(tour_points[:, 0], tour_points[:, 1], '-', label='Route', alpha=0.7)
    for i, (x, y) in enumerate(positions):
        ax.text(x, y + 0.5, f'{i}', fontsize=8, ha='center')
    ax.legend()
    ax.grid(True)
    plt.suptitle(f"CW vs RL Solution Comparison\n(Cap: {vehicle_capacity}, Batt: {battery_capacity})")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'cw_vs_rl_comparison_{vehicle_capacity}_{battery_capacity}.png')
    plt.close()
    debug_print(f"Side-by-side CW vs RL plot saved as 'cw_vs_rl_comparison_{vehicle_capacity}_{battery_capacity}.png'")

def try_plot_cw_vs_rl():
    # Try to load both C&W and PPO results and plot if both exist
    cw_results = None
    rl_results = None
    # Try to load C&W results from variable if available
    if 'cw_routes' in locals():
        # Convert list of lists to list of dictionaries for compatibility
        cw_results = [{'route': route} for route in cw_routes]
    # Try to load PPO results from file
    rl_routes = []
    try:
        with open("ppo_results.txt", "r") as f:
            for line in f:
                if line.startswith("PPO Route"):
                    route_str = line.split(":")[1].split(", Demand")[0].strip()
                    route = eval(route_str)
                    rl_routes.append({'route': route})
    except Exception:
        pass
    if cw_results and rl_routes:
        plot_cw_vs_rl(cw_results=cw_results, rl_results=rl_routes, positions=original_positions, charging_stations=charging_stations, vehicle_capacity=vehicle_capacity, battery_capacity=battery_capacity, num_customers=num_customers)

# Try to plot comparison after each run
try_plot_cw_vs_rl()
# --- End of Plotting Section ---
# --- RL Agent Only Plot Section ---
def plot_rl_only(positions, charging_stations, vehicle_capacity, battery_capacity, num_customers):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_title("RL Agent Solution Only")
    ax.scatter(positions[1:num_customers+1, 0], positions[1:num_customers+1, 1], c='blue', label='Customers')
    ax.scatter(positions[0, 0], positions[0, 1], c='red', marker='^', label='Depot')
    if charging_stations:
        ax.scatter(positions[charging_stations, 0], positions[charging_stations, 1], c='green', marker='s', label='Charging Station')
    # Parse RL routes from file
    rl_routes = []
    try:
        with open("evrp_charging_results.txt", "r") as f:
            for line in f:
                if line.startswith("RL Route"):
                    route_str = line.split(":")[1].split(", Demand")[0].strip()
                    route = eval(route_str)
                    rl_routes.append(route)
    except Exception:
        pass
    for route in rl_routes:
        tour_points = positions[route]
        ax.plot(tour_points[:, 0], tour_points[:, 1], '-', label='Route', alpha=0.7)
    for i, (x, y) in enumerate(positions):
        ax.text(x, y + 0.5, f'{i}', fontsize=8, ha='center')
    ax.legend()
    ax.grid(True)
    plt.suptitle(f"RL Agent Solution\n(Cap: {vehicle_capacity}, Batt: {battery_capacity})")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'rl_agent_solution_{vehicle_capacity}_{battery_capacity}.png')
    plt.show()
    debug_print(f"RL agent only plot saved as 'rl_agent_solution_{vehicle_capacity}_{battery_capacity}.png' and displayed.")
# --- Comparison Table Section ---
import re
def extract_metric(lines, key):
    for line in lines:
        if key in line:
            return line.split(':',1)[1].strip().split()[0]
    return None

def extract_float(lines, key):
    val = extract_metric(lines, key)
    try:
        return float(val)
    except:
        return None


def write_comparison_table():
    try:
        with open("cw_results.txt", "r") as f:
            cw_lines = f.readlines()
        with open("ppo_results.txt", "r") as f:
            rl_lines = f.readlines()
        metrics = [
            ("Total Vehicles", "Total Vehicles"),
            ("Total Demand Served", "Total Demand Served"),
            ("Total Time", "Total Time"),
            ("Total Battery Consumed", "Total Battery Consumed"),
        ]
        table = ["\n=== CW vs RL Comparison Table ===\n",
                 f"{'Metric':<25} | {'Clarke & Wright':<20} | {'RL Agent':<20}\n",
                 f"{'-'*25}|{'-'*20}|{'-'*20}\n"]
        for label, key in metrics:
            cw_val = extract_metric(cw_lines, key)
            rl_val = extract_metric(rl_lines, key)
            table.append(f"{label:<25} | {cw_val or '-':<20} | {rl_val or '-':<20}\n")
        rl_reward = extract_metric(rl_lines, "Total Reward")
        table.append(f"{'Total Reward (RL)':<25} | {'-':<20} | {rl_reward or '-':<20}\n")
        # Save comparison table to a new file
        with open("cw_vs_rl_comparison.txt", "w") as f:
            f.write("".join(table))
        # Print the table to the console for the user
        print("\n" + "".join(table))
        print("Comparison table saved to cw_vs_rl_comparison.txt and printed above.")
    except Exception as e:
        print(f"[Comparison Table Error] {e}")

# Automatically write comparison table if both results exist
import os
if os.path.exists("cw_results.txt") and os.path.exists("ppo_results.txt"):
    write_comparison_table()
    # Always plot RL agent only if PPO results exist
    with open("ppo_results.txt", "r") as f:
        content = f.read()
    if "RL Route" in content:
        plot_rl_only(original_positions, charging_stations, vehicle_capacity, battery_capacity, num_customers)

# [OPTIONAL] Expand the RL state for better learning
# Example: concatenate current position, battery, and a visited mask
# This is a placeholder; adapt as needed for your EVEnvironment
# Example:
# visited_mask = np.zeros(len(simple_positions))
# state = np.concatenate([state, [env.battery], visited_mask])
# state_dim = state.shape[0]
#
# For now, keep the dynamic hidden_dim logic for small state_dim

