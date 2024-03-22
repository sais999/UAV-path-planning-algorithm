import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.ndimage import gaussian_filter


def calculate_histogram_grid(area, cell_size=20, sigma=2):
    """
    Calculate the histogram grid for the given area.

    Parameters:
    - area: 2D numpy array representing the environment.
    - cell_size: Size of each cell in the histogram grid.
    - sigma: Standard deviation for Gaussian smoothing.

    Returns:
    - histogram_grid: 2D numpy array representing the obstacle density.
    """
    downsampled_area = area[::cell_size, ::cell_size]
    smoothed_area = gaussian_filter(downsampled_area.astype(float), sigma=sigma)
    histogram_grid = smoothed_area
    return histogram_grid


def calculate_safe_direction(histogram_grid, target_direction, opening_threshold=0.1):
    """
    Calculate a safe direction based on the histogram grid.

    Parameters:
    - histogram_grid: 2D numpy array representing the obstacle density.
    - target_direction: Desired direction towards the target.
    - opening_threshold: Threshold to consider a direction as safe.

    Returns:
    - safe_direction: The chosen safe direction to move towards.
    """
    # Simplified version: find the direction with the lowest obstacle density
    directions = np.linspace(0, 2 * np.pi, histogram_grid.size, endpoint=False)
    safe_directions = directions[histogram_grid.ravel() < opening_threshold]
    if safe_directions.size > 0:
        # Choose the direction closest to the target direction
        direction_diff = np.abs(safe_directions - target_direction)
        safe_direction = safe_directions[np.argmin(direction_diff)]
    else:
        safe_direction = target_direction  # No safe direction found, proceed with caution
    return safe_direction


def vfh_navigation(start, goal, area, cell_size=20):
    """
    Navigate from start to goal using the VFH algorithm.

    Parameters:
    - start: Starting position (x, y).
    - goal: Goal position (x, y).
    - area: 2D numpy array representing the environment.
    - cell_size: Size of each cell in the histogram grid.

    Returns:
    - path: List of positions representing the path from start to goal.
    """
    path = [start]
    current_position = np.array(start)
    goal_position = np.array(goal)

    while np.linalg.norm(current_position - goal_position) > cell_size:
        histogram_grid = calculate_histogram_grid(area, cell_size=cell_size)
        target_direction = np.arctan2(goal_position[1] - current_position[1], goal_position[0] - current_position[0])
        safe_direction = calculate_safe_direction(histogram_grid, target_direction)

        # Move in the safe direction
        step = np.array([np.cos(safe_direction), np.sin(safe_direction)]) * cell_size
        current_position += step
        path.append(tuple(current_position.astype(int)))

    return path


# Assuming the environment 'area' is already generated
path = vfh_navigation(start_point, end_point, area, cell_size=20)

# Visualization
display_array_with_graph_and_path(area, path, start_point, end_point)
