import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def display_array_with_graph(array_2d, graph_nodes, graph_edges):
    cmap = plt.cm.colors.ListedColormap(['white', 'black', 'red'])
    bounds = [0, 1, 2, 3]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    plt.imshow(array_2d, cmap=cmap, norm=norm, interpolation='none')

    # Plot graph nodes
    for node in graph_nodes:
        plt.plot(node[1], node[0], 'go', markersize=5)  # Green dots for graph nodes

    # Draw graph edges with weights
    for edge in graph_edges:
        plt.plot(edge[:, 1], edge[:, 0], 'b-', linewidth=2)
        midpoint = np.mean(edge, axis=0)
        plt.text(midpoint[1], midpoint[0], f'{np.linalg.norm(edge[0] - edge[1]):.2f}', color='black',
                 fontsize=8, ha='center', va='center')

    # Show colorbar
    cbar = plt.colorbar(ticks=[0, 1, 2])
    cbar.set_label('Color', rotation=270, labelpad=15)

    plt.show()

def is_valid_edge(edge, area):
    # Check if the edge overlaps with obstacles
    for point in np.linspace(edge[0], edge[1], num=100):
        x, y = map(int, point)
        if area[x, y] == 1:
            return False
    return True

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

# Create a graph and add nodes
G = nx.Graph()
G.add_nodes_from(graph_nodes)

# Connect nodes based on distance and add weights
for i in range(len(graph_nodes)):
    for j in range(i + 1, len(graph_nodes)):
        node_i, node_j = graph_nodes[i], graph_nodes[j]
        distance = np.linalg.norm(np.array(node_i) - np.array(node_j))
        if distance < 20:  # Set a threshold for connecting nodes
            edge = np.array([node_i, node_j])
            if is_valid_edge(edge, area):
                G.add_edge(node_i, node_j, weight=distance)

# Extract nodes and edges for visualization
graph_nodes = list(G.nodes())
graph_edges = [np.array([edge[0], edge[1]]) for edge in G.edges()]

# Display the array with the graph nodes and edges
display_array_with_graph(area, graph_nodes, graph_edges)
