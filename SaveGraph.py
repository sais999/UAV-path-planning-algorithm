import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os

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

# Create a 100x100 area
area_size = 500
area = np.zeros((area_size, area_size), dtype=int)

# Add random rectangle obstacles
num_obstacles = 55
min_obstacle_size = 35
max_obstacle_size = 35
safety = 5

# Store the nodes in a list
graph_nodes = []
obstacleId = 1
listOfCoordinates = []
listObstacleWidth = []
listObstacleHeight = []
# for loop to create the random obstacles
for _ in range(num_obstacles):
    nodeCoordinates = {}
    obstacle_width = np.random.randint(min_obstacle_size + safety, max_obstacle_size + 1 + safety)
    obstacle_height = np.random.randint(min_obstacle_size + safety, max_obstacle_size + 1 + safety)

    listObstacleWidth.append(obstacle_width)
    listObstacleHeight.append(obstacle_height)

    x = np.random.randint(0, area_size - obstacle_width)
    y = np.random.randint(0, area_size - obstacle_height)
    nodeCoordinates['x'] = x
    nodeCoordinates['y'] = y
    nodeCoordinates['id'] = obstacleId
    obstacleId = obstacleId + 1
    listOfCoordinates.append(nodeCoordinates)


    area[x:x + obstacle_width, y:y + obstacle_height] = 1
    #area[x][y] = 2

# Set the start point to the top-left corner and end point to the bottom-right corner
start_point = (1, 1)
end_point = (area_size - 2, area_size - 2)


# Create a graph and add nodes
G = nx.Graph()
G.add_nodes_from(map(tuple, graph_nodes))  # Convert nodes to tuples

# Folder path to save graphs
folder_path = 'Graphs/Graph1' #SOSOS SET THE PATH
os.makedirs(folder_path, exist_ok=True)

# Save the graph
graph_file_path = os.path.join(folder_path, 'graph.gexf')
nx.write_gexf(G, graph_file_path)

# Save the list of coordinates
coordinates_file_path = os.path.join(folder_path, 'coordinates.txt')
with open(coordinates_file_path, 'w') as f:
    for coord in listOfCoordinates:
        f.write(f"x: {coord['x']}, y: {coord['y']}\n")

# Save the list of obstacle widths
area_size_file_path = os.path.join(folder_path,'area_size.txt')
with open(area_size_file_path, 'w') as f:
    f.write(f"{area_size}\n")

# Save the list of obstacle widths
obstacle_width_file_path = os.path.join(folder_path,'obstacle_width.txt')
with open(obstacle_width_file_path, 'w') as f:
    for width in listObstacleWidth:
        f.write(f"{width},\n")

#Save the list of Obstacle heights
obstacle_height_file_path = os.path.join(folder_path,'obstacle_height.txt')
with open(obstacle_height_file_path, 'w') as f:
    for height in listObstacleHeight:
        f.write(f"{height},\n")

#Save the list of Obstacle id's
obstacle_id_file_path = os.path.join(folder_path,'obstacle_id.txt')
with open(obstacle_id_file_path, 'w') as f:
    for id in listOfCoordinates:
        f.write(f"{id['id']},\n")
# Save the array to a file
print(listObstacleWidth)
print(listObstacleHeight)
print(listOfCoordinates)
np.save('Graphs/Graph1/area.npy', area) #SOSOS SET THE PATH
# Display the array with the graph nodes, edges, and the shortest path
display_array_with_graph_and_path(area, graph_nodes, start_point, end_point)

print(f"Graph and coordinates saved in {folder_path}")
