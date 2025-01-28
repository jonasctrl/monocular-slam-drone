from __future__ import absolute_import, division, print_function

import numpy as np
from math import tan, pi
import sys, os
from scipy.spatial.transform import Rotation as R
from numba import njit, float64, int64
from typing import Tuple, List, Optional

PI_DIV_360 = pi / 360

@njit(cache=True)
def in_bounds(voxels, x, y, z):
    """Check if coordinates are within voxel bounds."""
    return 0 <= x < voxels.shape[0] and 0 <= y < voxels.shape[1] and 0 <= z < voxels.shape[2]

@njit(cache=True)
def clamp(v, a, b):
    """Clamp value between a and b."""
    return max(a, min(v, b))

@njit(cache=True)
def bresenham3d_raycast(p1, p2, voxels):
    """Optimized 3D Bresenham's line algorithm with raycast."""
    x1, y1, z1 = int(p1[0]), int(p1[1]), int(p1[2])
    x2, y2, z2 = int(p2[0]), int(p2[1]), int(p2[2])
    
    dx, dy, dz = abs(x2 - x1), abs(y2 - y1), abs(z2 - z1)
    sx = 1 if x2 > x1 else -1
    sy = 1 if y2 > y1 else -1
    sz = 1 if z2 > z1 else -1
    
    path = []
    
    if dx >= dy and dx >= dz:  # X dominant
        err1, err2 = 2 * dy - dx, 2 * dz - dx
        while x1 != x2 and in_bounds(voxels, x1, y1, z1):
            path.append((x1, y1, z1, voxels[x1, y1, z1]))
            
            if err1 > 0:
                y1 += sy
                err1 -= 2 * dx
            if err2 > 0:
                z1 += sz
                err2 -= 2 * dx
            err1 += 2 * dy
            err2 += 2 * dz
            x1 += sx
    
    elif dy >= dx and dy >= dz:  # Y dominant
        err1, err2 = 2 * dx - dy, 2 * dz - dy
        while y1 != y2 and in_bounds(voxels, x1, y1, z1):
            path.append((x1, y1, z1, voxels[x1, y1, z1]))
            
            if err1 > 0:
                x1 += sx
                err1 -= 2 * dy
            if err2 > 0:
                z1 += sz
                err2 -= 2 * dy
            err1 += 2 * dx
            err2 += 2 * dz
            y1 += sy
    
    else:  # Z dominant
        err1, err2 = 2 * dx - dz, 2 * dy - dz
        while z1 != z2 and in_bounds(voxels, x1, y1, z1):
            path.append((x1, y1, z1, voxels[x1, y1, z1]))
            
            if err1 > 0:
                x1 += sx
                err1 -= 2 * dz
            if err2 > 0:
                y1 += sy
                err2 -= 2 * dz
            err1 += 2 * dx
            err2 += 2 * dy
            z1 += sz

    if in_bounds(voxels, x2, y2, z2):
        path.append((x2, y2, z2, voxels[x2, y2, z2]))

    return path

@njit(parallel=True)
def depth_img_to_pcd(img, skip, factor, cam_params=None, fov=None, max_depth=float("inf")):
    """Convert depth image to point cloud with parallel processing."""
    height, width = img.shape
    
    if cam_params is not None:
        fx, fy, cx, cy = cam_params
    elif fov is not None:
        fx = width / (2 * tan(fov * PI_DIV_360))
        fy = fx * height / width
        cx = width / 2
        cy = height / 2
    else:
        raise ValueError("'cam_params' or 'fov' must be specified")

    max_points = ((height // skip) * (width // skip))
    points = np.empty((max_points, 3), dtype=np.float64)
    point_count = 0

    v_range = np.arange(1, height, skip)
    u_range = np.arange(1, width, skip)
    
    for v in v_range:
        for u in u_range:
            z = img[v, u] / factor
            if z > 0 and z <= max_depth:
                points[point_count] = [
                    (u - cx) * z / fx,
                    (v - cy) * z / fy,
                    z
                ]
                point_count += 1

    return points[:point_count]

def quaternion_from_two_vectors(v1, v2):
    """Calculate quaternion rotation between two vectors using vectorized operations."""
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    
    if v1_norm == 0 or v2_norm == 0:
        return np.array([0, 0, 0, 1])

    v1 = v1 / v1_norm
    v2 = v2 / v2_norm
    
    dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)
    
    if np.isclose(dot_product, 1.0):
        return np.array([0, 0, 0, 1])
    
    if np.isclose(dot_product, -1.0):
        orthogonal_axis = np.array([1, 0, 0]) if not np.allclose(v1, [1, 0, 0]) else np.array([0, 1, 0])
        axis = np.cross(v1, orthogonal_axis)
        axis = axis / np.linalg.norm(axis)
        return R.from_rotvec(np.pi * axis).as_quat()
    
    axis = np.cross(v1, v2)
    angle = np.arccos(dot_product)
    axis = axis / np.linalg.norm(axis)
    
    return R.from_rotvec(angle * axis).as_quat()
