import numpy as np
import heapq

import nav_config as cfg

# Heuristic: Euclidean distance between two points
def heuristic(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def v_empty(v):
    return cfg.occup_unkn == v or cfg.occup_min <= v < cfg.occup_thr


# Get neighbors in a 3D grid
def get_neighbors(node, grid):
    neighbors = []
    x, y, z = node
    # Check all possible directions in a 3D grid (26 possible neighbors)
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == 0 and dy == 0 and dz == 0:
                    continue  # Skip the current node
                new_x, new_y, new_z = x + dx, y + dy, z + dz
                if 0 <= new_x < grid.shape[0] and 0 <= new_y < grid.shape[1] and 0 <= new_z < grid.shape[2]:
                    # if abs(dx) + abs(dy) + abs(dz) > 1:
                        # continue  # Skip diag
                    diag_num = abs(dx) + abs(dy) + abs(dz)
                    if diag_num == 3:
                        if abs(dx) + abs(dy) == 2:
                            if not v_empty(grid[x, y, new_z]) or \
                                not v_empty(grid[new_x, y, z]) or \
                                not v_empty(grid[x, y, new_z]) or \
                                not v_empty(grid[x, new_y, new_z]) or \
                                not v_empty(grid[new_x, y, new_z]) or \
                                not v_empty(grid[new_x, new_y, z]):
                                continue
                    elif diag_num == 2:
                        if abs(dx) + abs(dy) == 2:
                            if not v_empty(grid[new_x, y, new_z]) or \
                                    not v_empty(grid[x, new_y, new_z]):
                                continue
                        elif abs(dy) + abs(dz) == 2:
                            if not v_empty(grid[new_x, y, new_z]) or \
                                    not v_empty(grid[new_x, new_y, z]):
                                continue
                        else:
                            if not v_empty(grid[x, new_y, new_z]) or \
                                    not v_empty(grid[new_x, new_y, z]):
                                continue

                        
                    # if abs(dx) + abs(dy) + abs(dz) > 1:
                        # continue  # Skip diag

                    if v_empty(grid[new_x, new_y, new_z]):
                    # if cfg.occup_unkn == v or cfg.occup_min <= v < cfg.occup_thr:
                        neighbors.append((new_x, new_y, new_z))
    return neighbors

# A* algorithm for 3D space
def a_star_3d(grid, start, goal):
    # Priority queue for the open list (min-heap)
    open_list = []
    heapq.heappush(open_list, (0, start))  # (cost, node)
    
    # Dictionary to store the cost of the shortest path to each node
    g_cost = {start: 0}
    
    # Dictionary to store the path
    came_from = {start: None}
    
    # Dictionary to store the total estimated cost (f = g + h)
    f_cost = {start: heuristic(start, goal)}
    
    # iters = 0
    # max_iters = 50
    while open_list:
        # Get the node with the lowest f_cost
        current_f_cost, current_node = heapq.heappop(open_list)
        
        # If the goal is reached, reconstruct the path
        
        # iters += 1
        # if iters > max_iters or current_node == goal:
            # print(f"path of {iters} iters cost={current_f_cost}")
        if current_node == goal:
            path = []
            while current_node:
                path.append(current_node)
                current_node = came_from[current_node]
            return path[::-1]  # Returnreversed path (from start to goal)
        
        # Explore neighbors
        for neighbor in get_neighbors(current_node, grid):
            tentative_g_cost = g_cost[current_node] + 1  # Cost from start to neighbor (assuming uniform grid)

            if neighbor not in g_cost or tentative_g_cost < g_cost[neighbor]:
                # Update the cost and path
                g_cost[neighbor] = tentative_g_cost
                f_cost[neighbor] = tentative_g_cost + heuristic(neighbor, goal)
                came_from[neighbor] = current_node
                
                # Add the neighbor to the open list
                heapq.heappush(open_list, (f_cost[neighbor], neighbor))
    
    # Return empty list if no path is found
    return []