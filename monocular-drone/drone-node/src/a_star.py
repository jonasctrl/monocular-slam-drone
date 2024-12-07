import numpy as np
import heapq
from numba import njit

import nav_config as cfg

def_tol = (0,0)
tol = (0,0)
max_iters = cfg.max_a_star_iters


# Heuristic: Euclidean distance between two points
# @njit
def heuristic(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2), ord=None)


@njit
def v_empty(v):
    return cfg.occup_unkn == v or cfg.occup_min <= v < cfg.occup_thr

@njit
def is_inside(pt, bounds):
    return  0 < pt[0] < bounds[0]-1 and \
            0 < pt[1] < bounds[1]-1 and \
            0 <= pt[2] < bounds[2]-1

@njit
def skip_dangerous_blocks2(pt, grid):
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if not v_empty(grid[pt[0] + dx, pt[1] + dy, pt[2] + dz]):
                    return True
    return False


# Get neighbors in a 3D grid
@njit
def get_neighbors(node, grid, height_restr):
    neighbors = []
    x, y, z = node
    
    # Check all possible directions in a 3D grid (26 possible neighbors)
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == 0 and dy == 0 and dz == 0:
                    continue  # Skip the current node
                new_x, new_y, new_z = x + dx, y + dy, z + dz
                new_pt = (new_x, new_y, new_z)

                if new_z < height_restr[0] or new_z > height_restr[1]:
                    continue  # Skip too low or too high voxels

                if is_inside(new_pt, grid.shape):
                    if skip_dangerous_blocks2(new_pt, grid):
                        continue

                    if v_empty(grid[new_x, new_y, new_z]):
                        neighbors.append((new_x, new_y, new_z))
    return neighbors

# A* algorithm for 3D space
def a_star_3d(grid, start, goal):
    global max_iters, tol
    
    for pt in [start, goal]:
        for pd, gd in zip(pt, grid.shape):
            if pd < 1 or pd > gd - 2:
                print(f"{pt} is out of range {grid.shape}")
                return [], True

    # Priority queue for the open list (min-heap)
    open_list = []
    heapq.heappush(open_list, (0, start))  # (cost, node)
    
    # Dictionary to store the cost of the shortest path to each node
    g_cost = {start: 0}
    
    # Dictionary to store the path
    came_from = {start: None}
    
    # Dictionary to store the total estimated cost (f = g + h)
    f_cost = {start: heuristic(start, goal)}
    
    iters = 0
    while open_list:
        iters += 1
        # Get the node with the lowest f_cost
        current_f_cost, current_node = heapq.heappop(open_list)
        
        # If the goal is reached, reconstruct the path
        is_unf = iters > max_iters

        
        if current_node == goal or is_unf:
            if is_unf:
                # Increase maximum iteration count and allowed tolerances if
                # path planning was cut short
                max_iters += cfg.unf_max_iters_incr
                tol = (tol[0] + cfg.unf_neg_tol_incr, tol[1] + cfg.unf_pos_tol_incr)
            else:
                max_iters = cfg.max_a_star_iters
                tol = def_tol


            path = []
            while current_node:
                path.append(current_node)
                current_node = came_from[current_node]
            return path[::-1], is_unf   # Returnreversed path (from start to goal)
        
        for neighbor in get_neighbors(current_node, grid, tol):
            tentative_g_cost = g_cost[current_node] + 1 

            if neighbor not in g_cost or tentative_g_cost < g_cost[neighbor]:
                # Update the cost and path
                g_cost[neighbor] = tentative_g_cost
                f_cost[neighbor] = tentative_g_cost + heuristic(neighbor, goal)
                came_from[neighbor] = current_node
                
                # Add the neighbor to the open list
                heapq.heappush(open_list, (f_cost[neighbor], neighbor))
    
    return [], True


def a_star_setup(tolerances):
    global tol, def_tol
    tol = tolerances
    def_tol = tolerances




