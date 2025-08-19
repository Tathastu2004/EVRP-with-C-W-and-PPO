import torch
import numpy as np
import matplotlib.pyplot as plt
from mp_utils import PPOPolicy, EVEnvironment, clarke_wright_evrp
from evaluate import load_evrp_tsp

def evaluate_ppo_model(model_path, test_file='test_case_evrp.tsp'):
    """
    Evaluate a trained PPO model on the test case.
    
    :param str model_path: Path to the trained PPO model
    :param str test_file: Path to the test case file
    """
    # Load test case
    (num_customers, positions, demands, charging_stations, vehicle_capacity, 
     battery_capacity, consumption_rate, min_soc, charging_time, max_vehicles, 
     positive_demand_customers) = load_evrp_tsp(test_file)
    
    print(f"[INFO] Evaluating PPO model on {test_file}")
    print(f"[INFO] Number of customers: {num_customers}")
    print(f"[INFO] Number of nodes: {len(positions)}")
    
    # --- Adaptive model loading to handle dimension mismatch ---
    raw_state = torch.load(model_path, map_location='cpu')
    if not isinstance(raw_state, dict):
        print(f"[ERROR] Checkpoint format unsupported: {type(raw_state)}")
        return None
    # Infer original network dimensions from weights
    if 'fc1.weight' not in raw_state or 'policy_head.weight' not in raw_state:
        print("[ERROR] State dict missing expected layers (fc1.weight/policy_head.weight)")
        return None
    pretrained_state_dim = raw_state['fc1.weight'].shape[1]
    pretrained_hidden = raw_state['fc1.weight'].shape[0]
    pretrained_action_dim = raw_state['policy_head.weight'].shape[0]
    inferred_total_nodes = (pretrained_state_dim - 1) // 2  # from formula 2*nodes + 1
    print(f"[ADAPT] Pretrained: state_dim={pretrained_state_dim}, action_dim={pretrained_action_dim}, total_nodes(incl depot)={inferred_total_nodes}")

    current_total_nodes = len(positions)  # includes depot
    if current_total_nodes != inferred_total_nodes:
        if current_total_nodes > inferred_total_nodes:
            print(f"[ADAPT][WARN] Test case has {current_total_nodes} nodes but model was trained on {inferred_total_nodes}. Truncating to match.")
            # Truncate positions/demands/charging stations/positive_demand_customers
            positions = positions[:inferred_total_nodes]
            demands = demands[:inferred_total_nodes]
            charging_stations = [cs for cs in charging_stations if cs < inferred_total_nodes]
            positive_demand_customers = [c for c in positive_demand_customers if c < inferred_total_nodes]
        else:
            print(f"[ADAPT][ERROR] Model expects more nodes ({inferred_total_nodes}) than provided ({current_total_nodes}). Cannot safely pad. Abort.")
            return None

    # Build policy with pretrained dims
    policy = PPOPolicy(pretrained_state_dim, pretrained_action_dim, dropout_rate=0.1)
    try:
        missing, unexpected = policy.load_state_dict(raw_state, strict=False)
        if missing:
            print(f"[ADAPT][INFO] Missing keys (ignored): {missing}")
        if unexpected:
            print(f"[ADAPT][INFO] Unexpected keys (ignored): {unexpected}")
        print(f"[INFO] Successfully loaded PPO model from {model_path}")
    except Exception as e:
        print(f"[ERROR] Failed to load PPO model after adaptation: {e}")
        return None
    
    policy.eval()
    
    # Create environment
    env = EVEnvironment(positions, charging_stations, battery_capacity=battery_capacity, 
                       consumption_rate=consumption_rate, min_soc=min_soc, 
                       charging_time=charging_time, max_vehicles=max_vehicles)
    
    # Evaluate PPO
    print("\n=== PPO Evaluation ===")
    ppo_results = evaluate_policy(policy, env, positions, demands, charging_stations, positive_demand_customers)
    
    # Evaluate C&W for comparison
    print("\n=== Clarke & Wright Evaluation ===")
    cw_results = evaluate_cw(positions, demands, charging_stations, vehicle_capacity, 
                           battery_capacity, consumption_rate, min_soc, charging_time, max_vehicles, positive_demand_customers)
    
    # Compare results
    print("\n=== Comparison ===")
    compare_results(ppo_results, cw_results)
    
    return ppo_results, cw_results

def evaluate_policy(policy, env, positions, demands, charging_stations, positive_demand_customers):
    """Evaluate PPO policy on the environment."""
    env.reset()
    
    total_reward = 0
    total_distance = 0
    customers_visited = 0
    route = [0]  # Start at depot
    current_node = 0
    
    print(f"Starting PPO evaluation from depot (node 0)")
    
    while True:
        # Get current state
        visited_mask = np.array(env.visited)
        battery = env.battery
        current_node_one_hot = np.zeros(len(positions))
        current_node_one_hot[current_node] = 1.0
        state = np.concatenate([current_node_one_hot, [battery], visited_mask])
        
        # Get action mask
        mask = env.get_action_mask()
        mask = np.asarray(mask).flatten()
        
        # Get policy action (deterministic for evaluation)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            mask_tensor = torch.FloatTensor(mask).unsqueeze(0)
            probs, value = policy(state_tensor, action_mask=mask_tensor)
            action = torch.argmax(probs, dim=-1).item()
        
        # Take step
        next_state, reward, done, info = env.step(action)
        
        # Update metrics
        total_reward += reward
        if hasattr(env, 'positions') and current_node < len(env.positions) and action < len(env.positions):
            distance = np.linalg.norm(env.positions[current_node] - env.positions[action])
            total_distance += distance
        
        route.append(action)
        current_node = action
        
        if action in positive_demand_customers:
            customers_visited += 1
        
        print(f"  Step: {len(route)-1}, Action: {action}, Reward: {reward:.2f}, Distance: {distance:.2f if 'distance' in locals() else 0:.2f}")
        
        if done:
            break
    
    results = {
        'total_reward': total_reward,
        'total_distance': total_distance,
        'customers_visited': customers_visited,
        'route': route,
        'total_customers': len(positive_demand_customers)
    }
    
    print(f"PPO Results:")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Total Distance: {total_distance:.2f}")
    print(f"  Customers Visited: {customers_visited}/{len(positive_demand_customers)}")
    print(f"  Route: {route}")
    
    return results

def evaluate_cw(positions, demands, charging_stations, vehicle_capacity, battery_capacity, 
               consumption_rate, min_soc, charging_time, max_vehicles, positive_demand_customers):
    """Evaluate Clarke & Wright algorithm."""
    try:
        routes, strict_vehicle_limit = clarke_wright_evrp(
            positions, demands, 0, charging_stations, vehicle_capacity, 
            battery_capacity, consumption_rate, min_soc, charging_time, max_vehicles
        )
        
        total_distance = 0
        customers_visited = 0
        
        for route in routes:
            # Calculate distance for this route
            for i in range(len(route) - 1):
                distance = np.linalg.norm(positions[route[i]] - positions[route[i+1]])
                total_distance += distance
            
            # Count customers visited
            for node in route:
                if node in positive_demand_customers:
                    customers_visited += 1
        
        results = {
            'total_distance': total_distance,
            'customers_visited': customers_visited,
            'routes': routes,
            'total_customers': len(positive_demand_customers),
            'num_vehicles': len(routes)
        }
        
        print(f"C&W Results:")
        print(f"  Total Distance: {total_distance:.2f}")
        print(f"  Customers Visited: {customers_visited}/{len(positive_demand_customers)}")
        print(f"  Number of Vehicles: {len(routes)}")
        print(f"  Routes: {routes}")
        
        return results
    
    except Exception as e:
        print(f"[ERROR] C&W evaluation failed: {e}")
        return None

def compare_results(ppo_results, cw_results):
    """Compare PPO and C&W results."""
    if ppo_results is None or cw_results is None:
        print("Cannot compare results - one or both evaluations failed")
        return
    
    print("\n=== Performance Comparison ===")
    print(f"{'Metric':<20} {'PPO':<15} {'C&W':<15} {'Difference':<15}")
    print("-" * 65)
    
    # Distance comparison
    ppo_distance = ppo_results['total_distance']
    cw_distance = cw_results['total_distance']
    distance_diff = ppo_distance - cw_distance
    print(f"{'Total Distance':<20} {ppo_distance:<15.2f} {cw_distance:<15.2f} {distance_diff:<15.2f}")
    
    # Customer coverage comparison
    ppo_customers = ppo_results['customers_visited']
    cw_customers = cw_results['customers_visited']
    customer_diff = ppo_customers - cw_customers
    print(f"{'Customers Visited':<20} {ppo_customers:<15} {cw_customers:<15} {customer_diff:<15}")
    
    # Coverage percentage
    total_customers = ppo_results['total_customers']
    ppo_coverage = (ppo_customers / total_customers) * 100
    cw_coverage = (cw_customers / total_customers) * 100
    coverage_diff = ppo_coverage - cw_coverage
    print(f"{'Coverage %':<20} {ppo_coverage:<15.1f} {cw_coverage:<15.1f} {coverage_diff:<15.1f}")
    
    # Efficiency (distance per customer)
    if ppo_customers > 0:
        ppo_efficiency = ppo_distance / ppo_customers
    else:
        ppo_efficiency = float('inf')
    
    if cw_customers > 0:
        cw_efficiency = cw_distance / cw_customers
    else:
        cw_efficiency = float('inf')
    
    efficiency_diff = ppo_efficiency - cw_efficiency
    print(f"{'Distance/Customer':<20} {ppo_efficiency:<15.2f} {cw_efficiency:<15.2f} {efficiency_diff:<15.2f}")
    
    # Summary
    print("\n=== Summary ===")
    if distance_diff < 0:
        print("✅ PPO achieved shorter total distance")
    else:
        print("❌ C&W achieved shorter total distance")
    
    if customer_diff > 0:
        print("✅ PPO visited more customers")
    elif customer_diff < 0:
        print("❌ C&W visited more customers")
    else:
        print("✅ Both algorithms visited the same number of customers")
    
    if coverage_diff > 0:
        print("✅ PPO achieved better customer coverage")
    else:
        print("❌ C&W achieved better customer coverage")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate PPO model on EVRP")
    parser.add_argument("--model_path", type=str, default="trained_ppo_policy.pth", 
                       help="Path to trained PPO model")
    parser.add_argument("--test_file", type=str, default="test_case_evrp.tsp", 
                       help="Path to test case file")
    
    args = parser.parse_args()
    
    evaluate_ppo_model(args.model_path, args.test_file)
