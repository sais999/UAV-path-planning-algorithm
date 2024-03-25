import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os

def display_array_with_graph_and_path(array_2d):
    cmap = plt.cm.colors.ListedColormap(['white', 'black'])
    bounds = [0, 1 ]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    plt.imshow(array_2d, cmap=cmap, norm=norm, interpolation='none')


    # Show colorbar
    cbar = plt.colorbar(ticks=[0, 1, 2])
    cbar.set_label('Color', rotation=270, labelpad=15)

    plt.show()

# Create a 100x100 area
area_size = 500
area = np.zeros((area_size, area_size), dtype=int)

# Add random rectangle obstacles
num_obstacles = 50
min_obstacle_size = 50
max_obstacle_size = 50
safety = 5

# Store the nodes in a list
graph_nodes = []
obstacleId = 1
listOfObstacles = []
# for loop to create the random obstacles
for _ in range(num_obstacles):
    obstacleInfo = {}
    obstacle_width = np.random.randint(min_obstacle_size + safety, max_obstacle_size + 1 + safety)
    obstacle_height = np.random.randint(min_obstacle_size + safety, max_obstacle_size + 1 + safety)

    x = np.random.randint(0, area_size - obstacle_width)
    y = np.random.randint(0, area_size - obstacle_height)
    obstacleInfo['x'] = x
    obstacleInfo['y'] = y
    obstacleInfo['id'] = obstacleId
    obstacleInfo['width'] = obstacle_width
    obstacleInfo['height'] = obstacle_height

    listOfObstacles.append(obstacleInfo)

    area[x:x + obstacle_width, y:y + obstacle_height] = obstacleId
    obstacleId = obstacleId + 1
    #area[x][y] = 2

# Set the start point to the top-left corner and end point to the bottom-right corner
start_point = (1, 1)
end_point = (area_size - 2, area_size - 2)


# Create a graph and add nodes
G = nx.Graph()
G.add_nodes_from(map(tuple, graph_nodes))  # Convert nodes to tuples

# Folder path to save graphs
folder_path = 'Graphs/Graph1'  #SOSOS SET THE PATH
os.makedirs(folder_path, exist_ok=True)

# Save the graph
graph_file_path = os.path.join(folder_path, 'graph.gexf')
nx.write_gexf(G, graph_file_path)

# Save the list of coordinates
obstacle_info_file_path = os.path.join(folder_path, 'obstacles.txt')
with open(obstacle_info_file_path, 'w') as f:
    for obstacle in listOfObstacles:
        f.write(f"x: {obstacle['x']}, y: {obstacle['y']}, id: {obstacle['id']}, width: {obstacle['width']}, height: {obstacle['height']}\n")

# Save the list of obstacle widths
area_size_file_path = os.path.join(folder_path,'area_size.txt')
with open(area_size_file_path, 'w') as f:
    f.write(f"{area_size}\n")


print(listOfObstacles)
np.save('Graphs/Graph1/area.npy', area) #SOSOS SET THE PATH
# Display the array with the graph nodes, edges, and the shortest path
display_array_with_graph_and_path(area)

print(f"Graph and coordinates saved in {folder_path}")
