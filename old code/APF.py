import numpy as np
import matplotlib.pyplot as plt

def compute_attractive_potential(x, y, goal, scale=1.0):
    """
    Compute the attractive potential at a point.
    """
    dist = np.sqrt((x - goal[0])**2 + (y - goal[1])**2)
    return 0.5 * scale * dist**2
def display_array_with_graph_and_path(array_2d, graph_nodes, start_point, end_point):
    """
    Displays the environment, obstacles, start and end points, and the path.

    Parameters:
    - array_2d: The environment represented as a 2D numpy array.
    - graph_nodes: List of tuples representing the path nodes.
    - start_point: Tuple representing the starting point (x, y).
    - end_point: Tuple representing the end point (x, y).
    """
    cmap = plt.cm.colors.ListedColormap(['white', 'black', 'red'])
    bounds = [0, 1, 2, 3]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    plt.imshow(array_2d, cmap=cmap, norm=norm, interpolation='none')

    # Plot graph nodes as green dots
    for node in graph_nodes:
        plt.plot(node[1], node[0], 'go', markersize=5)

    # Plot start and end points
    plt.plot(start_point[1], start_point[0], 'bo', markersize=8, label='Start (A)')
    plt.plot(end_point[1], end_point[0], 'yo', markersize=8, label='End (B)')

    plt.legend()
    plt.colorbar(ticks=[0, 1, 2]).set_label('Color', rotation=270, labelpad=15)
    plt.show()
def compute_repulsive_potential(x, y, obstacles, influence_distance=50, scale=1.0):
    """
    Compute the repulsive potential for a point from all obstacles.
    """
    rep_potential = 0
    for obstacle in obstacles:
        dist = np.sqrt((x - obstacle[0])**2 + (y - obstacle[1])**2) - obstacle[2]
        if dist < influence_distance:
            if dist <= 0:
                # Inside or very close to the obstacle
                rep_potential += 0.5 * scale * influence_distance**2
            else:
                rep_potential += 0.5 * scale * (1/dist - 1/influence_distance)**2
    return rep_potential

def compute_total_potential(x, y, goal, obstacles, attractive_scale=1.0, repulsive_scale=0.01, influence_distance=50):
    """
    Compute the total potential at a point by summing attractive and repulsive potentials.
    """
    att_potential = compute_attractive_potential(x, y, goal, scale=attractive_scale)
    rep_potential = compute_repulsive_potential(x, y, obstacles, influence_distance, scale=repulsive_scale)
    return att_potential + rep_potential


def gradient_descent(start, goal, obstacles, step_size=1.0, max_steps=1000, attractive_scale=1.0, repulsive_scale=0.1,
                     influence_distance=100):
    """
    Navigate from start to goal using gradient descent on the total potential field.
    Parameters adjusted for stronger obstacle avoidance.
    """
    current_position = np.array(start, dtype=float)
    path = [start]

    for _ in range(max_steps):
        # Calculate gradients for attractive and repulsive potentials separately
        gradient_att_x = compute_attractive_potential(current_position[0] + step_size, current_position[1], goal,
                                                      scale=attractive_scale) - \
                         compute_attractive_potential(current_position[0], current_position[1], goal,
                                                      scale=attractive_scale)
        gradient_att_y = compute_attractive_potential(current_position[0], current_position[1] + step_size, goal,
                                                      scale=attractive_scale) - \
                         compute_attractive_potential(current_position[0], current_position[1], goal,
                                                      scale=attractive_scale)

        gradient_rep_x = compute_repulsive_potential(current_position[0] + step_size, current_position[1], obstacles,
                                                     influence_distance, scale=repulsive_scale) - \
                         compute_repulsive_potential(current_position[0], current_position[1], obstacles,
                                                     influence_distance, scale=repulsive_scale)
        gradient_rep_y = compute_repulsive_potential(current_position[0], current_position[1] + step_size, obstacles,
                                                     influence_distance, scale=repulsive_scale) - \
                         compute_repulsive_potential(current_position[0], current_position[1], obstacles,
                                                     influence_distance, scale=repulsive_scale)
        # Combine gradients
        gradient_x = gradient_att_x + gradient_rep_x
        gradient_y = gradient_att_y + gradient_rep_y
        gradient = np.array([gradient_x, gradient_y])

        # Compute the gradient magnitude
        gradient_magnitude = np.linalg.norm(gradient)
        # Normalize to prevent overly large steps
        if gradient_magnitude > 0:
            dynamic_step_size = step_size
            current_position -= (dynamic_step_size * gradient) / gradient_magnitude
        else:
            # If gradient magnitude is zero, break the loop to avoid infinite loop
            break

        path.append(tuple(current_position.astype(int)))
        if np.linalg.norm(current_position - goal) < step_size:
            break

    return path


# Read the graph from graph.gexf
graph_file_path = '../Graphs/Graph1/graph.gexf'
#G = nx.read_gexf(graph_file_path)
num_obstacles = 25
file_path = '../Graphs/Graph1/area_size.txt'
# Read the file and extract obstacle heights
with open(file_path, 'r') as f:
    for line in f:
        # Remove trailing comma and convert to integer
        area_size = int(line.strip())

area = np.zeros((area_size, area_size), dtype=int)
graph_nodes = []


# File path
file_path = 'Graphs/Graph1/coordinates.txt'

# List to store coordinates
listOfCoordinates = []

# Read the file and extract coordinates
with open(file_path, 'r') as f:
    for line in f:
        # Split the line into parts
        parts = line.strip().split(', ')

        # Ensure the line has the expected format
        if len(parts) == 2:
            # Extract x and y values
            x = int(parts[0][2:])
            y = int(parts[1][2:])

            # Create a dictionary for each coordinate
            nodeCoordinates = {'x': x, 'y': y}

            # Append to the list
            listOfCoordinates.append(nodeCoordinates)
# File path
file_path = 'Graphs/Graph1/obstacle_height.txt'

# List to store obstacle heights
listObstacleHeight = []

# Read the file and extract obstacle heights
with open(file_path, 'r') as f:
    for line in f:
        # Remove trailing comma and convert to integer
        obstacle_height = int(line.strip().rstrip(','))

        # Append to the list
        listObstacleHeight.append(obstacle_height)
# Load the array back from the file
area = np.load('../Graphs/Graph1/area.npy')
# Print the list of obstacle heights
print(listObstacleHeight)

# Print the list of coordinates
print(listOfCoordinates)
# File path
file_path = 'Graphs/Graph1/obstacle_width.txt'

# List to store obstacle heights
listObstacleWidth = []

# Read the file and extract obstacle heights
with open(file_path, 'r') as f:
    for line in f:
        # Remove trailing comma and convert to integer
        obstacle_width = int(line.strip().rstrip(','))

        # Append to the list
        listObstacleWidth.append(obstacle_width)

# Print the list of obstacle heights
print(listObstacleWidth)
start_point = (1, 1)
end_point = (area_size - 2, area_size - 2)
# Example Usage:
area = np.load('../Graphs/Graph1/area.npy')  # Load the obstacle array
obstacles = [(coord['x'], coord['y'], max(width, height)/2) for coord, width, height in zip(listOfCoordinates, listObstacleWidth, listObstacleHeight)]  # Approximating obstacles as circles for simplicity
path = gradient_descent(start_point, end_point, obstacles)

# Display the result
display_array_with_graph_and_path(area, path, start_point, end_point)
