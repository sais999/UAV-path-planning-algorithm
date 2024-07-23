import numpy as np
import networkx as nx
import os
import matplotlib.pyplot as plt
import time
#function to display graph, obstacles, nodes, start and end points, and shortest path
def display_array_with_graph_and_path(array_2d, start_point, end_point, path):
    cmap = plt.cm.colors.ListedColormap(['white', 'black'])
    bounds = [0, 1]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    plt.imshow(array_2d, cmap=cmap, norm=norm, interpolation='none')

    # # Plot graph nodes
    # for node in graph_nodes:
    #     plt.plot(node[1], node[0], 'go', markersize=5)  # Green dots for graph nodes

    # Plot start and end points
    plt.plot(start_point[1], start_point[0], 'bo', markersize=8, label='Start (A)')
    plt.plot(end_point[1], end_point[0], 'yo', markersize=8, label='End (B)')

    # Plot the path
    if path:
        path_nodes = np.array(path)
        plt.plot(path_nodes[:, 1], path_nodes[:, 0], 'r-', linewidth=2, label='Shortest Path (A to B)')

    # Show legend
    #plt.legend()

    # Show colorbar
    cbar = plt.colorbar(ticks=[0, 1, 2])
    cbar.set_label('Color', rotation=270, labelpad=15)

    plt.show()

#function to check if an edge is on the available space
def is_valid_edge(edge, area):
    # Check if the edge overlaps with obstacles
    for point in np.linspace(edge[0], edge[1], num=100):
        x, y = map(int, point)
        if area[x, y] != 0:
            return False
    return True

 # Read the graph from graph.gexf
graph_file_path = 'Graphs/Graph1/graph.gexf'  #SOSOS SET THE PATH
G = nx.read_gexf(graph_file_path)
#num_obstacles = 25
# Create the area
file_path = 'Graphs/Graph1/area_size.txt'
# Read the file and extract area size
with open(file_path, 'r') as f:
    for line in f:

        area_size = int(line.strip())

area = np.zeros((area_size, area_size), dtype=int)
graph_nodes = []

# File path
file_path = 'Graphs/Graph1/obstacles.txt'  # Update the path if necessary

# List to store obstacles
listOfObstacles = []

# Read the file and extract obstacle data
with open(file_path, 'r') as f:
    for line in f:
        # Split the line into parts
        parts = line.strip().split(', ')
        obstacle = {}

        # Process each part of the line to extract obstacle data
        for part in parts:
            key, value = part.split(": ")
            obstacle[key] = int(value)  # Convert values to integers

        # Append the obstacle dictionary to the list
        listOfObstacles.append(obstacle)

# The listOfObstacles now contains dictionaries with full obstacle data
print(listOfObstacles)


# Load the array back from the file
area = np.load('Graphs/Graph1/area.npy') #SOSOS SET THE PATH
print(area_size)

start_time = time.time()
# Identify corners and store them as nodes
counter = 0
for obstacle in listOfObstacles:
     x = obstacle['x']
     y = obstacle['y']
     obstacle_width_for = obstacle['width']
     obstacle_height_for = obstacle['height']
     counter = counter + 1
     if area[x-1][y-1] == 0:
         graph_nodes.append((x-1, y-1))
     if area[x-1][y+obstacle_height_for] == 0:
         graph_nodes.append((x-1, y+obstacle_height_for))
     if area[x+obstacle_width_for][y-1] == 0:
         graph_nodes.append((x+obstacle_width_for, y-1))
     if area[x+obstacle_width_for][y+obstacle_height_for] == 0:
         graph_nodes.append((x+obstacle_width_for, y+obstacle_height_for))

# Get available space nodes (outside obstacles)
#available_space_nodes = [(i, j) for i in range(1, area_size - 1) for j in range(1, area_size - 1) if area[i, j] == 0]

# Select random start (A) and end (B) points from available space nodes
#start_point = np.random.choice(np.arange(len(available_space_nodes)), size=2, replace=False)
#end_point = np.random.choice(np.arange(len(available_space_nodes)), size=2, replace=False)
#start_point = available_space_nodes[start_point]
#end_point = available_space_nodes[end_point]
# Set the start point to the top-left corner and end point to the bottom-right corner
#start_point = (1, 1)
#end_point = (area_size - 2, area_size - 2)
start_point = (area_size/2, 1)
end_point = (area_size/2, area_size - 2)

G.add_nodes_from(map(tuple, graph_nodes))  # Convert nodes to tuples
# Add start and end points to the graph
G.add_node(start_point)
G.add_node(end_point)

graph_nodes.append(start_point)
graph_nodes.append(end_point)
# set threshold for the distance of a possible edge
threshold = 2 * area_size
# Connect nodes based on distance and add weights
for i, node in enumerate(graph_nodes):
    for j in range(i+1, len(graph_nodes)):
        other_node = graph_nodes[j]
        distance = np.linalg.norm(np.array(node) - np.array(other_node))
        edge = np.array([node, other_node])
        # set threshold for the distance of a possible edge
        if distance < threshold and is_valid_edge(edge, area):
            G.add_edge(tuple(node), tuple(other_node), weight=distance)  # Convert nodes to tuples



# Find the shortest path using Dijkstra's algorithm
shortest_path = None
try:
    shortest_path = nx.shortest_path(G, source=start_point, target=end_point, weight='weight')
    # Calculate the length of the shortest path
    shortest_path_length = nx.shortest_path_length(G, source=start_point, target=end_point, weight='weight')
    print(f"Length of the shortest path: {shortest_path_length}")
except nx.NetworkXNoPath:
    print("No valid path found. Please try again with different obstacle distribution.")
end_time = time.time()
# Calculate the total time taken
total_time = end_time - start_time
print(f"Total execution time: {total_time} seconds")
# Extract nodes and edges for visualization
graph_nodes = list(G.nodes())
graph_edges = [np.array([np.array(edge[0]), np.array(edge[1])]) for edge in G.edges()]

# Display the array with the graph nodes, edges, and the shortest path
display_array_with_graph_and_path(area, start_point, end_point, shortest_path)
