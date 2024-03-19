import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def display_array_with_graph(array_2d, graph_nodes):
    cmap = plt.cm.colors.ListedColormap(['white', 'black', 'red'])
    bounds = [0, 1, 2, 3]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    plt.imshow(array_2d, cmap=cmap, norm=norm, interpolation='none')

    # Plot graph nodes
    for node in graph_nodes:
        plt.plot(node[1], node[0], 'go', markersize=5)  # Green dots for graph nodes

    # Show colorbar
    cbar = plt.colorbar(ticks=[0, 1, 2])
    cbar.set_label('Color', rotation=270, labelpad=15)

    plt.show()

# Create a 100x100 area
area_size = 100
area = np.zeros((area_size, area_size), dtype=int)

# Add random rectangle obstacles
num_obstacles = 20
min_obstacle_size = 5
max_obstacle_size = 15

# Store the nodes in a list
graph_nodes = []

for _ in range(num_obstacles):
    obstacle_width = np.random.randint(min_obstacle_size, max_obstacle_size + 1)
    obstacle_height = np.random.randint(min_obstacle_size, max_obstacle_size + 1)

    x = np.random.randint(0, area_size - obstacle_width)
    y = np.random.randint(0, area_size - obstacle_height)

    # Identify corners and store them as nodes

    if area[x-1][y-1] == 0:
        graph_nodes.append((x-1, y-1))
    if area[x-1][y+obstacle_height] == 0:
        graph_nodes.append((x-1, y+obstacle_height))
    if area[x+obstacle_width][y-1] == 0:
        graph_nodes.append((x+obstacle_width, y-1))
    if area[x+obstacle_width][y+obstacle_height] == 0:
        graph_nodes.append((x+obstacle_width, y+obstacle_height))

    area[x:x + obstacle_width, y:y + obstacle_height] = 1

# Display the array with the graph nodes
display_array_with_graph(area, graph_nodes)
