import numpy as np
import networkx as nx
import os
# Read the graph from graph.gexf
#graph_file_path = 'Graphs/graph.gexf'
#G = nx.read_gexf(graph_file_path)
num_obstacles = 25
# Create a 100x100 area
area_size = 100
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
for i in range(num_obstacles):
     x = listOfCoordinates[i]['x']
     y = listOfCoordinates[i]['y']
     obstacle_width = listObstacleWidth[i]
     obstacle_height = listObstacleHeight[i]
     if area[x-1][y-1] == 0:
         graph_nodes.append((x-1, y-1))
     if area[x-1][y+obstacle_height] == 0:
         graph_nodes.append((x-1, y+obstacle_height))
     if area[x+obstacle_width][y-1] == 0:
         graph_nodes.append((x+obstacle_width, y-1))
     if area[x+obstacle_width][y+obstacle_height] == 0:
         graph_nodes.append((x+obstacle_width, y+obstacle_height))
