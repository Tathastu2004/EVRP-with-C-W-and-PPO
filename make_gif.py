import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
from matplotlib.patches import Circle, FancyBboxPatch
import matplotlib.patches as mpatches

def make_gif_from_tour(positions, tour_sequence, gif_path, fps=2, dpi=100):
    """
    Create an animated GIF from a tour sequence.
    
    Args:
        positions: numpy array of node positions (n_nodes, 2)
        tour_sequence: list of lists, each representing a tour step
        gif_path: path to save the GIF
        fps: frames per second for the animation
        dpi: dots per inch for the output
    """
    try:
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Set up the plot
        ax.set_xlim(positions[:, 0].min() - 1, positions[:, 0].max() + 1)
        ax.set_ylim(positions[:, 1].min() - 1, positions[:, 1].max() + 1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Plot all nodes
        ax.scatter(positions[:, 0], positions[:, 1], c='lightblue', s=100, zorder=5)
        
        # Plot depot (node 0) with different color
        ax.scatter(positions[0, 0], positions[0, 1], c='red', s=150, marker='^', zorder=6, label='Depot')
        
        # Add node labels
        for i, (x, y) in enumerate(positions):
            ax.text(x, y + 0.3, str(i), fontsize=8, ha='center', va='bottom', fontweight='bold')
        
        # Initialize empty line for the tour
        line, = ax.plot([], [], 'b-', linewidth=2, alpha=0.7)
        point, = ax.plot([], [], 'ro', markersize=8)
        
        def animate(frame):
            if frame < len(tour_sequence):
                tour = tour_sequence[frame]
                if len(tour) > 1:
                    # Get coordinates for the current tour
                    tour_coords = positions[tour]
                    line.set_data(tour_coords[:, 0], tour_coords[:, 1])
                    
                    # Highlight current position
                    if tour:
                        current_pos = positions[tour[-1]]
                        point.set_data([current_pos[0]], [current_pos[1]])
                    else:
                        point.set_data([], [])
                else:
                    line.set_data([], [])
                    point.set_data([], [])
            else:
                line.set_data([], [])
                point.set_data([], [])
            
            return line, point
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=len(tour_sequence), 
                                     interval=1000//fps, blit=True, repeat=True)
        
        # Save as GIF
        # Ensure the data directory exists
        os.makedirs(os.path.dirname(gif_path), exist_ok=True)
        
        # Save the animation
        anim.save(gif_path, writer='pillow', fps=fps, dpi=dpi)
        plt.close(fig)
        
        print(f"[GIF] Animation saved to {gif_path}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to create GIF: {e}")
        return False

def make_gif_from_tour_advanced(positions, tour_sequence, gif_path, charging_stations=None, 
                               vehicle_capacity=None, battery_capacity=None, fps=2, dpi=100):
    """
    Create an advanced animated GIF with charging stations and additional information.
    
    Args:
        positions: numpy array of node positions (n_nodes, 2)
        tour_sequence: list of lists, each representing a tour step
        gif_path: path to save the GIF
        charging_stations: list of charging station indices
        vehicle_capacity: vehicle capacity for display
        battery_capacity: battery capacity for display
        fps: frames per second for the animation
        dpi: dots per inch for the output
    """
    try:
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Set up the plot
        ax.set_xlim(positions[:, 0].min() - 1, positions[:, 0].max() + 1)
        ax.set_ylim(positions[:, 1].min() - 1, positions[:, 1].max() + 1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Plot all nodes
        ax.scatter(positions[:, 0], positions[:, 1], c='lightblue', s=100, zorder=5)
        
        # Plot depot (node 0) with different color
        ax.scatter(positions[0, 0], positions[0, 1], c='red', s=150, marker='^', zorder=6, label='Depot')
        
        # Plot charging stations if provided
        if charging_stations:
            charging_positions = positions[charging_stations]
            ax.scatter(charging_positions[:, 0], charging_positions[:, 1], 
                      c='green', s=120, marker='s', zorder=6, label='Charging Station')
        
        # Add node labels
        for i, (x, y) in enumerate(positions):
            ax.text(x, y + 0.3, str(i), fontsize=8, ha='center', va='bottom', fontweight='bold')
        
        # Initialize empty line for the tour
        line, = ax.plot([], [], 'b-', linewidth=2, alpha=0.7)
        point, = ax.plot([], [], 'ro', markersize=8)
        
        # Add title with information
        title_text = f"EVRP Tour Animation"
        if vehicle_capacity:
            title_text += f" | Vehicle Capacity: {vehicle_capacity}"
        if battery_capacity:
            title_text += f" | Battery: {battery_capacity}"
        ax.set_title(title_text, fontsize=12, fontweight='bold')
        
        def animate(frame):
            if frame < len(tour_sequence):
                tour = tour_sequence[frame]
                if len(tour) > 1:
                    # Get coordinates for the current tour
                    tour_coords = positions[tour]
                    line.set_data(tour_coords[:, 0], tour_coords[:, 1])
                    
                    # Highlight current position
                    if tour:
                        current_pos = positions[tour[-1]]
                        point.set_data([current_pos[0]], [current_pos[1]])
                    else:
                        point.set_data([], [])
                else:
                    line.set_data([], [])
                    point.set_data([], [])
            else:
                line.set_data([], [])
                point.set_data([], [])
            
            return line, point
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=len(tour_sequence), 
                                     interval=1000//fps, blit=True, repeat=True)
        
        # Save as GIF
        # Ensure the data directory exists
        os.makedirs(os.path.dirname(gif_path), exist_ok=True)
        
        # Save the animation
        anim.save(gif_path, writer='pillow', fps=fps, dpi=dpi)
        plt.close(fig)
        
        print(f"[GIF] Advanced animation saved to {gif_path}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to create advanced GIF: {e}")
        return False

if __name__ == "__main__":
    # Example usage
    print("make_gif.py - GIF generation utility for EVRP tours")
    print("Use make_gif_from_tour() or make_gif_from_tour_advanced() functions")
