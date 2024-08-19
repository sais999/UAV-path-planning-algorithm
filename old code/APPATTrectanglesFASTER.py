import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time

def display_array_with_graph_and_path(array_2d, start_point, end_point, graph_nodes, path=None):
    cmap = plt.cm.colors.ListedColormap(['white', 'black'])
    bounds = [0, 1]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    plt.imshow(array_2d, cmap=cmap, norm=norm, interpolation='none')

    if graph_nodes:
        graph_nodes = np.array(graph_nodes)
        plt.plot(graph_nodes[:, 1], graph_nodes[:, 0], 'go', markersize=5)

    plt.plot(start_point[1], start_point[0], 'bo', markersize=8, label='Start (A)')
    plt.plot(end_point[1], end_point[0], 'yo', markersize=8, label='End (B)')

    if path:
        path_nodes = np.array(path)
        plt.plot(path_nodes[:, 1], path_nodes[:, 0], 'r-', linewidth=2, label='Shortest Path (A to B)')

    plt.legend()
    cbar = plt.colorbar(ticks=[0, 1])
    cbar.set_label('Color', rotation=270, labelpad=15)
    plt.show()

def get_obstacle_collided_id(edge, area):
    points = np.linspace(edge[0], edge[1], num=100).astype(int)
    collision_ids = area[points[:, 0], points[:, 1]]
    valid_collisions = collision_ids[collision_ids != 0]
    if valid_collisions.size > 0:
        return valid_collisions[0]
    return -1

def is_valid_edge(edge, area):
    points = np.linspace(edge[0], edge[1], num=100).astype(int)
    return np.all(area[points[:, 0], points[:, 1]] == 0)

def get_temp_nodes(obstacle, area):
    x, y, w, h = obstacle['x'], obstacle['y'], obstacle['width'], obstacle['height']
    temp_nodes = []
    if area[x - 1, y - 1] == 0:
        temp_nodes.append((x - 1, y - 1))
    if area[x - 1, y + h] == 0:
        temp_nodes.append((x - 1, y + h))
    if area[x + w, y - 1] == 0:
        temp_nodes.append((x + w, y - 1))
    if area[x + w, y + h] == 0:
        temp_nodes.append((x + w, y + h))
    return temp_nodes

def add_node(node, unique_graph_nodes_set, graph_nodes):
    if node not in unique_graph_nodes_set:
        unique_graph_nodes_set.add(node)
        graph_nodes.append(node)

def add_new_obstacle_found(obstacle, unique_obstacles_found_set, obstacles_found):
    if obstacle['id'] not in unique_obstacles_found_set:
        unique_obstacles_found_set.add(obstacle['id'])
        obstacles_found.append(obstacle)

# Read the graph from graph.gexf
graph_file_path = 'Graphs/Graph1/graph.gexf'
G = nx.read_gexf(graph_file_path)

# Read area size
with open('Graphs/Graph1/area_size.txt', 'r') as f:
    area_size = int(f.read().strip())

area = np.load('Graphs/Graph1/area.npy')

# Read obstacles
listOfObstacles = []
with open('Graphs/Graph1/obstacles.txt', 'r') as f:
    for line in f:
        obstacle = {k: int(v) for k, v in (part.split(": ") for part in line.strip().split(', '))}
        listOfObstacles.append(obstacle)

unique_graph_nodes_set = set()
unique_obstacles_found_set = set()
graph_nodes = []
obstacles_found = []

# Set start and end points
start_point = (area_size // 2, 1)
end_point = (area_size // 2, area_size - 2)

add_node(start_point, unique_graph_nodes_set, graph_nodes)
G.add_node(start_point)
G.add_node(end_point)

start_time = time.time()
pathCreated = False

while not pathCreated:
    new_nodes = []
    for node in graph_nodes:
        edge_front = (node, end_point)
        edge_back = (node, start_point)
        if get_obstacle_collided_id(edge_front, area) == -1:
            pathCreated = True
            break
        else:
            if get_obstacle_collided_id(edge_back, area) != -1:
                obstacle_back = next((ob for ob in listOfObstacles if ob['id'] == get_obstacle_collided_id(edge_back, area)), None)
                if obstacle_back and obstacle_back['id'] not in unique_obstacles_found_set:
                    temp_nodes = get_temp_nodes(obstacle_back, area)
                    new_nodes.extend(temp_nodes)
                    add_new_obstacle_found(obstacle_back, unique_obstacles_found_set, obstacles_found)
            obstacle_front = next((ob for ob in listOfObstacles if ob['id'] == get_obstacle_collided_id(edge_front, area)), None)
            if obstacle_front and obstacle_front['id'] not in unique_obstacles_found_set:
                temp_nodes = get_temp_nodes(obstacle_front, area)
                new_nodes.extend(temp_nodes)
                add_new_obstacle_found(obstacle_front, unique_obstacles_found_set, obstacles_found)

    for new_node in new_nodes:
        add_node(new_node, unique_graph_nodes_set, graph_nodes)

add_node(end_point, unique_graph_nodes_set, graph_nodes)

threshold = area_size * 2
graph_nodes_array = np.array(graph_nodes)

for i, node in enumerate(graph_nodes):
    distances = np.linalg.norm(graph_nodes_array - node, axis=1)
    close_indices = np.where(distances < threshold)[0]
    for j in close_indices:
        if j > i:
            other_node = graph_nodes[j]
            edge = (node, tuple(other_node))
            if is_valid_edge(edge, area):
                G.add_edge(node, tuple(other_node), weight=distances[j])

shortest_path = None
try:
    shortest_path = nx.shortest_path(G, source=start_point, target=end_point, weight='weight')
    shortest_path_length = nx.shortest_path_length(G, source=start_point, target=end_point, weight='weight')
    print(f"Length of the shortest path: {shortest_path_length}")
except nx.NetworkXNoPath:
    print("No valid path found. Please try again with different obstacle distribution.")

end_time = time.time()
print(f"Total execution time: {end_time - start_time} seconds")
print(f"Total obstacles found: {len(obstacles_found)}")
display_array_with_graph_and_path(area, start_point, end_point, graph_nodes, shortest_path)
