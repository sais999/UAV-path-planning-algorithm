import numpy as np
import matplotlib.pyplot as plt
def display_array(array_2d):
    cmap = plt.cm.colors.ListedColormap(['white', 'black', 'red'])
    bounds = [0, 1, 2, 3]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    plt.imshow(array_2d, cmap=cmap, norm=norm, interpolation='none')

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

for _ in range(num_obstacles):
    obstacle_width = np.random.randint(min_obstacle_size, max_obstacle_size + 1)
    obstacle_height = np.random.randint(min_obstacle_size, max_obstacle_size + 1)

    x = np.random.randint(0, area_size - obstacle_width)
    y = np.random.randint(0, area_size - obstacle_height)

    area[x:x + obstacle_width, y:y + obstacle_height] = 1
for i in range(area_size):
    for j in range(area_size):
        if(area[i][j] == 1):
            if(area[i-1][j] == 0 and area[i][j-1] == 0):
                area[i][j] = 2
            if (area[i + 1][j] == 0 and area[i][j - 1] == 0):
                area[i][j] = 2
            if (area[i + 1][j] == 0 and area[i][j + 1] == 0):
                area[i][j] = 2
            if (area[i - 1][j] == 0 and area[i][j + 1] == 0):
                area[i][j] = 2



display_array(area)
