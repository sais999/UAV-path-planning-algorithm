import numpy as np
import networkx as nx
import os
import matplotlib.pyplot as plt
def display_array_with_graph_and_path(array_2d, graph_nodes, start_point, end_point):
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

    # Show legend
    plt.legend()

    # Show colorbar
    cbar = plt.colorbar(ticks=[0, 1, 2])
    cbar.set_label('Color', rotation=270, labelpad=15)

    plt.show()

# Read the graph from graph.gexf
graph_file_path = '../Graphs/Graph1/graph.gexf'
G = nx.read_gexf(graph_file_path)
num_obstacles = 25
file_path = '../Graphs/Graph1/area_size.txt'
# Read the file and extract obstacle heights
with open(file_path, 'r') as f:
    for line in f:
        # Remove trailing comma and convert to integer
        area_size = int(line.strip())

area = np.zeros((area_size, area_size), dtype=int)
graph_nodes = []
# # Get the maximum x and y values to determine the size of the area
# #max_x = max(int(node[0]) for node in G.nodes())
# #max_y = max(int(node[1]) for node in G.nodes())
#
# # Create a 2D array (area) based on the maximum x and y values
# #area_size = max(max_x, max_y) + 3  # Add a margin
# #area = np.zeros((area_size, area_size), dtype=int)
#
# # Fill the area based on the graph nodes
# for node in G.nodes():
#     area[int(node[0]), int(node[1])] = 1
#
# # Print the area
# print("Area:")
# print(area)

# File path
file_path = '../Graphs/Graph1/coordinates.txt'

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
file_path = '../Graphs/Graph1/obstacle_height.txt'

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
file_path = '../Graphs/Graph1/obstacle_width.txt'

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
for coord in listOfCoordinates:
     x = coord['x']
     y = coord['y']
     obstacle_width_for = listObstacleWidth[i]
     obstacle_height_for = listObstacleHeight[i]
     i = i + 1
     if area[x-1][y-1] == 0:
         graph_nodes.append((x-1, y-1))
     if area[x-1][y+obstacle_height_for] == 0:
         graph_nodes.append((x-1, y+obstacle_height_for))
     if area[x+obstacle_width_for][y-1] == 0:
         graph_nodes.append((x+obstacle_width_for, y-1))
     if area[x+obstacle_width_for][y+obstacle_height_for] == 0:
         graph_nodes.append((x+obstacle_width_for, y+obstacle_height_for))



# Set the start point to the top-left corner and end point to the bottom-right corner
start_point = (1, 1)
end_point = (area_size - 2, area_size - 2)
G.add_nodes_from(map(tuple, graph_nodes))  # Convert nodes to tuples
# Display the array with the graph nodes, edges, and the shortest path
display_array_with_graph_and_path(area, graph_nodes, start_point, end_point)