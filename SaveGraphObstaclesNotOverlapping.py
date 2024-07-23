import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os

def display_array_with_graph_and_path(array_2d):
    cmap = plt.cm.colors.ListedColormap(['white', 'black'])
    bounds = [0, 1]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    plt.imshow(array_2d, cmap=cmap, norm=norm, interpolation='none')
    cbar = plt.colorbar(ticks=[0, 1])
    cbar.set_label('Color', rotation=270, labelpad=15)
    plt.show()

def is_overlapping(new_obstacle, obstacles):
    for obstacle in obstacles:
        if not (new_obstacle['x'] + new_obstacle['width'] < obstacle['x'] or
                new_obstacle['x'] > obstacle['x'] + obstacle['width'] or
                new_obstacle['y'] + new_obstacle['height'] < obstacle['y'] or
                new_obstacle['y'] > obstacle['y'] + obstacle['height']):
            return True
    return False

max_attempts_per_obstacle = 100  # Maximum attempts to place each obstacle

# Create a 500x500 area
area_size = 5000
area = np.zeros((area_size, area_size), dtype=int)

# Add random rectangle obstacles
num_obstacles = 600
min_obstacle_size = 50
max_obstacle_size = 3300
safety_margin = 5
listOfObstacles = []

for _ in range(num_obstacles):
    for attempt in range(max_attempts_per_obstacle):
        obstacle_width = np.random.randint(min_obstacle_size, max_obstacle_size + 1) + safety_margin
        obstacle_height = np.random.randint(min_obstacle_size, max_obstacle_size + 1) + safety_margin
        x = np.random.randint(0, area_size - obstacle_width)
        y = np.random.randint(0, area_size - obstacle_height)

        new_obstacle = {
            'x': x,
            'y': y,
            'width': obstacle_width ,
            'height': obstacle_height ,
            'id': len(listOfObstacles) + 1
        }

        if not is_overlapping(new_obstacle, listOfObstacles):
            listOfObstacles.append(new_obstacle)
            area[x:x+obstacle_width, y:y+obstacle_height] = new_obstacle['id']

            break
    else:
        print(f"Could not place obstacle {_} after {max_attempts_per_obstacle} attempts.")
print(f"Number of obstacles is {len(listOfObstacles)}")

# Display the area with obstacles
display_array_with_graph_and_path(area)

# Create a graph and add nodes
G = nx.Graph()

# Folder path to save graphs and other data
folder_path = 'Graphs/Graph1'
os.makedirs(folder_path, exist_ok=True)

# Save the graph (even though it's empty here, for demonstration)
graph_file_path = os.path.join(folder_path, 'graph.gexf')
nx.write_gexf(G, graph_file_path)

# Save the list of coordinates
obstacle_info_file_path = os.path.join(folder_path, 'obstacles.txt')
with open(obstacle_info_file_path, 'w') as f:
    for obstacle in listOfObstacles:
        f.write(f"x: {obstacle['x']}, y: {obstacle['y']}, id: {obstacle['id']}, width: {obstacle['width']}, height: {obstacle['height']}\n")

# Save the area size
area_size_file_path = os.path.join(folder_path,'area_size.txt')
with open(area_size_file_path, 'w') as f:
    f.write(f"{area_size}\n")

# Save the array to a file
np.save(os.path.join(folder_path, 'area.npy'), area)

print(f"Graph and coordinates saved in {folder_path}")
