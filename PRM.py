import numpy as np
import networkx as nx
import os
import matplotlib.pyplot as plt
import time
def display_array_with_graph_and_path(array_2d, graph_nodes, start_point, end_point, path):
    cmap = plt.cm.colors.ListedColormap(['white', 'black', 'red'])
    bounds = [0, 1, 2, 3]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    plt.imshow(array_2d, cmap=cmap, norm=norm, interpolation='none')

    # Plot graph nodes
    for node in graph_nodes:
        plt.plot(node[1], node[0], 'go', markersize=5)  # Green dots for graph nodes

    # Plot start and end points
    plt.plot(start_point[1], start_point[0], 'bo', markersize=8, label='Start (A)')
    plt.plot(end_point[1], end_point[0], 'yo', markersize=8, label='End (B)')

    # Plot the path
    if path:
        path_nodes = np.array(path)
        plt.plot(path_nodes[:, 1], path_nodes[:, 0], 'r-', linewidth=2, label='Shortest Path (A to B)')

    # Show legend
    plt.legend()

    # Show colorbar
    cbar = plt.colorbar(ticks=[0, 1, 2])
    cbar.set_label('Color', rotation=270, labelpad=15)

    plt.show()

def is_valid_edge(edge, area,):  # Reduced from 500 to 100
    for point in np.linspace(edge[0], edge[1], num=100):
        x, y = map(int, point)
        if area[x, y] == 1:
            return False
    return True
def get_closest_nodes(current_node, all_nodes, n):
    distances = [np.linalg.norm(np.array(current_node) - np.array(node)) for node in all_nodes]
    sorted_indices = np.argsort(distances)
    return [all_nodes[i] for i in sorted_indices[:n]]

# Read the graph from graph.gexf
graph_file_path = 'Graphs/Graph1/graph.gexf'
G = nx.read_gexf(graph_file_path)
num_obstacles = 25
file_path = 'Graphs/Graph1/area_size.txt'
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
area = np.load('Graphs/Graph1/area.npy')
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
# Identify corners and store them as nodes
i = 0
# Get available space nodes (outside obstacles)
available_space_nodes = [(i, j) for i in range(1, area_size - 1) for j in range(1, area_size - 1) if area[i, j] == 0]


start_time = time.time()
# Set the start point to the top-left corner and end point to the bottom-right corner
start_point = (1, 1)
end_point = (area_size - 2, area_size - 2)
path = 0

num_nodes = 200 #set the number of nodes
#create random points
for i in range(num_nodes):
    random_index = np.random.choice(len(available_space_nodes))
    random_point = available_space_nodes[random_index]
    graph_nodes.append(random_point)


# Connect the edges to multiple nearby nodes
n_connections = num_nodes# Adjust based on desired connectivity
for random_node in graph_nodes:
    closest_random_nodes = get_closest_nodes(random_node, graph_nodes, n=n_connections)
    for target_node in closest_random_nodes:
        edge = (random_node, target_node)
        if is_valid_edge(edge, area):
            G.add_edge(tuple(random_node), tuple(target_node),
                       weight=np.linalg.norm(np.array(random_node) - np.array(target_node)))
# Add start and end points to the graph
G.add_node(start_point)
G.add_node(end_point)

# Connect start and end points to the graph
closest_start_nodes = get_closest_nodes(start_point, graph_nodes, n_connections)
closest_end_nodes = get_closest_nodes(end_point, graph_nodes, n_connections)

G.add_edge(start_point, tuple(closest_start_nodes[0]), weight=np.linalg.norm(np.array(start_point) - np.array(closest_start_nodes[0])))
G.add_edge(end_point, tuple(closest_end_nodes[0]), weight=np.linalg.norm(np.array(end_point) - np.array(closest_end_nodes[0])))

# Find the shortest path using Dijkstra's algorithm
shortest_path = None
try:
    shortest_path = nx.shortest_path(G, source=start_point, target=end_point, weight='weight')
    # Calculate the length of the shortest path
    shortest_path_length = nx.shortest_path_length(G, source=start_point, target=end_point,
                                                                   weight='weight')
    print(f"Length of the shortest path: {shortest_path_length}")
except nx.NetworkXNoPath:
    print("No valid path found. Please try again with different obstacle distribution.")
end_time = time.time()
# Calculate the total time taken
total_time = end_time - start_time
print(f"Total execution time: {total_time} seconds")
G.add_nodes_from(map(tuple, graph_nodes))  # Convert nodes to tuples
# Display the array with the graph nodes, edges, and the shortest path
display_array_with_graph_and_path(area, graph_nodes, start_point, end_point, shortest_path)
