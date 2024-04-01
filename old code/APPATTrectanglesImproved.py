import numpy as np
import networkx as nx
import os
import matplotlib.pyplot as plt
import time
def display_array_with_graph_and_path(array_2d, graph_nodes, start_point, end_point, path):
    cmap = plt.cm.colors.ListedColormap(['white', 'black'])
    bounds = [0, 1]
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
    #plt.legend()

    # Show colorbar
    cbar = plt.colorbar(ticks=[0, 1, 2])
    cbar.set_label('Color', rotation=270, labelpad=15)

    plt.show()
def display_array_with_graph_and_path_with_edges(array_2d, graph_nodes, start_point, end_point, path):
    cmap = plt.cm.colors.ListedColormap(['white', 'black'])
    bounds = [0, 1]
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

def get_obstacle_collided_id(edge, area):
    for point in np.linspace(edge[0], edge[1], num=100):
        x, y = map(int, point)
        if area[x, y] != 0:
            return area[x,y]
    return -1
def is_valid_edge(edge, area):
    # Check if the edge overlaps with obstacles
    for point in np.linspace(edge[0], edge[1], num=100):
        x, y = map(int, point)
        if area[x, y] != 0:
            return False
    return True
def get_obstacle(listOfObstacles, id):
    for obstacle in listOfObstacles:
        if obstacle['id']==id:
            return obstacle
def get_closest_nodes(current_node, all_nodes, n):
    distances = [np.linalg.norm(np.array(current_node) - np.array(node)) for node in all_nodes]
    sorted_indices = np.argsort(distances)
    return [all_nodes[i] for i in sorted_indices[:n]]

def get_temp_nodes(listOfObstacles, area):
    counter = 0
    graph_nodes = []
    for obstacle in listOfObstacles:
        x = obstacle['x']
        y = obstacle['y']
        obstacle_width_for = obstacle['width']
        obstacle_height_for = obstacle['height']
        counter = counter + 1
        if area[x - 1][y - 1] == 0:
            graph_nodes.append((x - 1, y - 1))
        if area[x - 1][y + obstacle_height_for] == 0:
            graph_nodes.append((x - 1, y + obstacle_height_for))
        if area[x + obstacle_width_for][y - 1] == 0:
            graph_nodes.append((x + obstacle_width_for, y - 1))
        if area[x + obstacle_width_for][y + obstacle_height_for] == 0:
            graph_nodes.append((x + obstacle_width_for, y + obstacle_height_for))
    return graph_nodes

def add_node_to_graph(node):
    if node not in unique_graph_nodes_set:
        unique_graph_nodes_set.add(node)
        graph_nodes.append(node)
def add_node_to_candidate(node):
    if node not in unique_candidate_nodes_set:
        unique_candidate_nodes_set.add(node)
        candidate_nodes_list.append(node)
def add_node_to_final(node):
    if node not in unique_final_nodes_set:
        unique_final_nodes_set.add(node)
        final_nodes_list.append(node)
def add_new_obstacle_found(obstacle):
    obstacle_id = obstacle['id']  # Use the unique ID of the obstacle
    if obstacle_id not in unique_obstacles_found_set:
        unique_obstacles_found_set.add(obstacle_id)
        obstacles_found.append(obstacle)
 # Read the graph from graph.gexf
graph_file_path = '../Graphs/Graph1/graph.gexf'  #SOSOS SET THE PATH
G = nx.read_gexf(graph_file_path)
#num_obstacles = 25
# Create the area
file_path = '../Graphs/Graph1/area_size.txt'
# Read the file and extract area size
with open(file_path, 'r') as f:
    for line in f:

        area_size = int(line.strip())

area = np.zeros((area_size, area_size), dtype=int)


# File path
file_path = '../Graphs/Graph1/obstacles.txt'  # Update the path if necessary

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
area = np.load('../Graphs/Graph1/area.npy') #SOSOS SET THE PATH
print(area_size)
unique_graph_nodes_set = set()
unique_final_nodes_set = set()
unique_candidate_nodes_set = set()
unique_obstacles_found_set = set()
graph_nodes = []
temp_nodes = []
obstacles_found = []
obstacles_found_id = []

# Identify corners and store them as nodes

# Get available space nodes (outside obstacles)
#vailable_space_nodes = [(i, j) for i in range(1, area_size - 1) for j in range(1, area_size - 1) if area[i, j] == 0]

# Select random start (A) and end (B) points from available space nodes
#start_point = np.random.choice(np.arange(len(available_space_nodes)), size=2, replace=False)
#end_point = np.random.choice(np.arange(len(available_space_nodes)), size=2, replace=False)
#start_point = available_space_nodes[start_point]
#end_point = available_space_nodes[end_point]
# Set the start point to the top-left corner and end point to the bottom-right corner
#start_point = (1, 1)
#end_point = (area_size - 2, area_size - 2)
start_point = (250, 1)
end_point = (250, area_size - 2)
G.add_nodes_from(map(tuple, graph_nodes))  # Convert nodes to tuples
# Add start and end points to the graph
G.add_node(start_point)
G.add_node(end_point)
nodes = []
final_nodes_list = []
candidate_nodes_list = []
final_nodes_list.append(start_point)
candidate_nodes_list.append(end_point)
#START OF THE ALGORITHM
add_node_to_graph(start_point)
#graph_nodes.append(end_point)
start_time = time.time()
obstacles_found_id_list = []
pathCreated = False
while pathCreated==False:
    if not candidate_nodes_list:  # Check if candidate_nodes_list is empty
        print("Candidate nodes list is empty. No path could be found.")
        break  # Exit the loop since there are no more nodes to process

    O = final_nodes_list[-1]  # No need to use len()-1, just -1 works
    #final_nodes_list.pop()
    O = start_point
    D = candidate_nodes_list[-1]
    edge = (O, D)
    print(edge)
    if D == end_point and get_obstacle_collided_id(edge, area)==-1: #einai to teliko simio kai den exei sigkrousi
        pathCreated = True
        print("i am in if 1")
        # for node in temp_nodes:
        #     add_node(node)
        # break
    elif D == end_point and get_obstacle_collided_id(edge, area)!=-1: #einai to teliko simio kai exei sigkrousi
        obstacles_collided_id = get_obstacle_collided_id(edge, area)
        print("i am in if 2")
        if obstacles_collided_id not in unique_obstacles_found_set:
            add_new_obstacle_found(get_obstacle(listOfObstacles, obstacles_collided_id))
            nodes = get_temp_nodes(obstacles_found, area)

            for node in nodes:
                add_node_to_candidate(node)
                add_node_to_graph(node)
            nodes.clear()
        else:
            add_node_to_final(D)
            #candidate_nodes_list.pop()
            print("This obstacle already exists")
    elif D != end_point and get_obstacle_collided_id(edge, area)==-1:#den einai to teliko simio kai den exei sigkrousi
        add_node_to_final(D)
        candidate_nodes_list.pop()
        print("i am in if 3")
    elif D!= end_point and get_obstacle_collided_id(edge,area)!= -1: # den einai to teliko simio kai exei sigkrousi
        obstacles_collided_id = get_obstacle_collided_id(edge, area)
        print("i am in if 4")
        if obstacles_collided_id not in unique_obstacles_found_set:
            add_new_obstacle_found(get_obstacle(listOfObstacles, obstacles_collided_id))
            nodes = get_temp_nodes(obstacles_found, area)

            for node in nodes:
                add_node_to_candidate(node)
                add_node_to_graph(node)
            nodes.clear()
        else:
            add_node_to_final(D)
            candidate_nodes_list.pop()
            print("This obstacle already exists")






add_node_to_graph(end_point)
threshold = area_size*2
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


graph_edges = [np.array([np.array(edge[0]), np.array(edge[1])]) for edge in G.edges()]
end_time = time.time()
total_time = end_time - start_time
print(f"Total execution time: {total_time} seconds")
display_array_with_graph_and_path(area, graph_nodes, start_point, end_point, shortest_path)
#Display the array with the graph nodes, edges, and the shortest path
#display_array_with_graph_and_path_with_edges(area, graph_nodes,graph_edges, start_point, end_point, shortest_path)




