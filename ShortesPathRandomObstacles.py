import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

#function to display graph, obstacles, nodes, start and end points, and shortest path
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
#function to check if an edge is on the available space
def is_valid_edge(edge, area):
    # Check if the edge overlaps with obstacles
    for point in np.linspace(edge[0], edge[1], num=5000):
        x, y = map(int, point)
        if area[x, y] == 1:
            return False
    return True

#def get_closest_nodes(current_node, all_nodes, n=2):
#    distances = [np.linalg.norm(np.array(current_node) - np.array(node)) for node in all_nodes]
#    sorted_indices = np.argsort(distances)
#    return [all_nodes[i] for i in sorted_indices[:n]]

# Create a 100x100 area
area_size = 200
area = np.zeros((area_size, area_size), dtype=int)

# Add random rectangle obstacles
num_obstacles = 45
min_obstacle_size = 5
max_obstacle_size = 20
safety = 2 #safety distance from every obstacle
# Store the nodes in a list
graph_nodes = []
#for loop to create the random obstacles
for _ in range(num_obstacles):
    obstacle_width = np.random.randint(min_obstacle_size + safety, max_obstacle_size + 1 + safety)
    obstacle_height = np.random.randint(min_obstacle_size + safety, max_obstacle_size + 1 + safety)

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

    #commented code is for safety to be shown with red color
    #area[x: x + obstacle_width + safety, y: y - safety] = 2
    #area[x: x - safety, y:y + obstacle_height + safety ]
    #area[x:x + obstacle_width + safety, y:y + obstacle_height + safety] = 2
    area[x:x + obstacle_width, y:y + obstacle_height] = 1


# Create a graph and add nodes
G = nx.Graph()
G.add_nodes_from(map(tuple, graph_nodes))  # Convert nodes to tuples
# Get available space nodes (outside obstacles)
available_space_nodes = [(i, j) for i in range(1, area_size - 1) for j in range(1, area_size - 1) if area[i, j] == 0]

# Select random start (A) and end (B) points from available space nodes
#start_point, end_point = np.random.choice(np.arange(len(available_space_nodes)), size=2, replace=False)
#start_point = available_space_nodes[start_point]
#end_point = available_space_nodes[end_point]
start_point = (0, 0)
end_point = (area_size-1, area_size-1)

# Add start and end points to the graph
G.add_node(start_point)
G.add_node(end_point)

graph_nodes.append(start_point)
graph_nodes.append(end_point)
# Connect start and end points to the graph
#closest_start_nodes = get_closest_nodes(start_point, graph_nodes, n=1)
#closest_end_nodes = get_closest_nodes(end_point, graph_nodes, n=1)

#G.add_edge(start_point, tuple(closest_start_nodes[0]), weight=np.linalg.norm(np.array(start_point) - np.array(closest_start_nodes[0])))
#G.add_edge(end_point, tuple(closest_end_nodes[0]), weight=np.linalg.norm(np.array(end_point) - np.array(closest_end_nodes[0])))

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
except nx.NetworkXNoPath:
    print("No valid path found. Please try again with different obstacle distribution.")

# Extract nodes and edges for visualization
graph_nodes = list(G.nodes())
graph_edges = [np.array([np.array(edge[0]), np.array(edge[1])]) for edge in G.edges()]

# Display the array with the graph nodes, edges, and the shortest path
display_array_with_graph_and_path(area, graph_nodes, start_point, end_point, shortest_path)

