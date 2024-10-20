
import numpy as np
from math import tan, pi


def bresenham3d_get_collision(p1, p2, voxels, bounds, empty_val=0):
    x1, y1, z1 = p1
    x2, y2, z2 = p2

    # Compute deltas
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    dz = abs(z2 - z1)
    
    # Determine the direction of the steps
    sx = 1 if x2 > x1 else -1
    sy = 1 if y2 > y1 else -1
    sz = 1 if z2 > z1 else -1
    
    # Initialize error terms
    if dx >= dy and dx >= dz:  # X is dominant
        err1 = 2 * dy - dx
        err2 = 2 * dz - dx
        while x1 != x2:
            if x1<bounds[0][0] or x1>=bounds[0][1] or y1<bounds[1][0] or y1>=bounds[1][1] or z1<bounds[2][0] or z1>=bounds[2][1]: 
                return None
            if voxels[x1, y1, z1] > empty_val:
                return (x1, y1, z1)
            if err1 > 0:
                y1 += sy
                err1 -= 2 * dx
            if err2 > 0:
                z1 += sz
                err2 -= 2 * dx
            err1 += 2 * dy
            err2 += 2 * dz
            x1 += sx
    elif dy >= dx and dy >= dz:  # Y is dominant
        err1 = 2 * dx - dy
        err2 = 2 * dz - dy
        while y1 != y2:
            if x1<bounds[0][0] or x1>=bounds[0][1] or y1<bounds[1][0] or y1>=bounds[1][1] or z1<bounds[2][0] or z1>=bounds[2][1]: 
                return None
            if voxels[x1, y1, z1] > empty_val:
                return (x1, y1, z1)
            if err1 > 0:
                x1 += sx
                err1 -= 2 * dy
            if err2 > 0:
                z1 += sz
                err2 -= 2 * dy
            err1 += 2 * dx
            err2 += 2 * dz
            y1 += sy
    else:  # Z is dominant
        err1 = 2 * dx - dz
        err2 = 2 * dy - dz
        while z1 != z2:
            if x1<bounds[0][0] or x1>=bounds[0][1] or y1<bounds[1][0] or y1>=bounds[1][1] or z1<bounds[2][0] or z1>=bounds[2][1]: 
                return None
            if voxels[x1, y1, z1] > empty_val:
                return (x1, y1, z1)
            if err1 > 0:
                x1 += sx
                err1 -= 2 * dz
            if err2 > 0:
                y1 += sy
                err2 -= 2 * dz
            err1 += 2 * dx
            err2 += 2 * dy
            z1 += sz
    
    return None
    # Add the final voxel (end point)
    # voxels.append((x2, y2, z2))
    
    # return voxels

def bresenham3d_check_known_space(p1, p2, voxels, empty_val=0):
    """
    3D Bresenham's line algorithm to generate voxel coordinates along a straight line.
    
    Parameters:
    - p1: tuple of (x1, y1, z1) representing the first point
    - p2: tuple of (x2, y2, z2) representing the second point
    
    Returns:
    - List of tuples representing the coordinates of voxels along the line.
    """
    x1, y1, z1 = p1
    x2, y2, z2 = p2

    # voxels = []
    
    # Compute deltas
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    dz = abs(z2 - z1)
    
    # Determine the direction of the steps
    sx = 1 if x2 > x1 else -1
    sy = 1 if y2 > y1 else -1
    sz = 1 if z2 > z1 else -1
    
    # Initialize error terms
    if dx >= dy and dx >= dz:  # X is dominant
        err1 = 2 * dy - dx
        err2 = 2 * dz - dx
        while x1 != x2:
            # voxels.append((x1, y1, z1))
            voxels[x1, y1, z1] = empty_val
            if err1 > 0:
                y1 += sy
                err1 -= 2 * dx
            if err2 > 0:
                z1 += sz
                err2 -= 2 * dx
            err1 += 2 * dy
            err2 += 2 * dz
            x1 += sx
    elif dy >= dx and dy >= dz:  # Y is dominant
        err1 = 2 * dx - dy
        err2 = 2 * dz - dy
        while y1 != y2:
            # voxels.append((x1, y1, z1))
            voxels[x1, y1, z1] = empty_val
            if err1 > 0:
                x1 += sx
                err1 -= 2 * dy
            if err2 > 0:
                z1 += sz
                err2 -= 2 * dy
            err1 += 2 * dx
            err2 += 2 * dz
            y1 += sy
    else:  # Z is dominant
        err1 = 2 * dx - dz
        err2 = 2 * dy - dz
        while z1 != z2:
            # voxels.append((x1, y1, z1))
            voxels[x1, y1, z1] = empty_val
            if err1 > 0:
                x1 += sx
                err1 -= 2 * dz
            if err2 > 0:
                y1 += sy
                err2 -= 2 * dz
            err1 += 2 * dx
            err2 += 2 * dy
            z1 += sz
    
    # Add the final voxel (end point)
    # voxels.append((x2, y2, z2))
    
    # return voxels

def depth_img_to_pcd(img, skip, factor, cam_params=None, fov=None, max_depth=float("inf")):
    height, width = img.shape
    point_cloud = []
    if cam_params is not None:
        (fx, fy, cx, cy) = cam_params
    elif fov is not None:
        fx = width / (2*tan(fov*pi/360))
        fy = fx * height / width
        cx = width / 2
        cy = height / 2
    else:
        raise Exception("'cam_params' or 'fov' must be specified ")

    for v in range(1, height, skip):
        for u in range(1, width, skip):
            z = img[v, u] / factor  # Depth value (in meters or millimeters)
            # if z == 0:  # Skip pixels with no depth
            if z == 0 or z > max_depth:  # Skip pixels with no depth
                continue

            # Convert (u, v, z) to (X, Y, Z)
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            point_cloud.append([x, y, z])

    return point_cloud
