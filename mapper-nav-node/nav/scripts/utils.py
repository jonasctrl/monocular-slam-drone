
import numpy as np
from math import tan, pi
import sys, os
from scipy.spatial.transform import Rotation as R
import nav_config as cfg

from numba import njit

@njit
def in_bounds(voxels, x, y, z):
    s=voxels.shape
    return 0 <= x < s[0] and 0 <= y < s[1] and 0 <= z < s[2] 



@njit
def clamp(v, a, b):
    if v < a:
        return a
    if v > b:
        return b
    return v


@njit
def bresenham3d_raycast(p1, p2, voxels):
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

    path = []
    
    # Initialize error terms
    if dx >= dy and dx >= dz:  # X is dominant
        err1 = 2 * dy - dx
        err2 = 2 * dz - dx
        while x1 != x2 and in_bounds(voxels, x1, y1, z1):
            path.append((x1, y1, z1, voxels[x1, y1, z1]))
            # voxels[x1, y1, z1] = empty_val
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
        while y1 != y2 and in_bounds(voxels, x1, y1, z1): 
            # path.append((x1, y1, z1))
            path.append((x1, y1, z1, voxels[x1, y1, z1]))
            # voxels[x1, y1, z1] = empty_val
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
        while z1 != z2 and in_bounds(voxels, x1, y1, z1):
            # path.append((x1, y1, z1))
            path.append((x1, y1, z1, voxels[x1, y1, z1]))
            # voxels[x1, y1, z1] = empty_val
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
    if in_bounds(voxels, x1, y1, z1):
        path.append((x2, y2, z2, voxels[x2, y2, z2]))

    return path

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


def generate_z_axis_quaternions(num_steps=10):
    """
    Generate a list of equally spaced quaternions representing rotations around the z-axis.
    
    :param num_steps: Number of equally spaced rotations (default is 10 for 360 degrees).
    :return: List of quaternions as (w, x, y, z).
    """
    quaternions = []
    # Divide 360 degrees into 'num_steps' equally spaced angles
    angles = np.linspace(0, 2 * np.pi, num_steps, endpoint=False)  # In radians
    
    for theta in angles:
        # Calculate quaternion for rotation around z-axis
        w = np.cos(theta / 2)
        x = 0
        y = 0
        z = np.sin(theta / 2)
        quaternions.append((w, x, y, z))
    
    return quaternions


def quaternion_from_two_vectors(v1, v2):
    # Normalize both vectors
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    # Calculate the axis of rotation (cross product)
    axis = np.cross(v1, v2)
    
    # Calculate the angle between the two vectors (dot product and arccos)
    angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))

    # If the vectors are parallel (angle is 0), return an identity quaternion
    if np.isclose(angle, 0):
        return R.from_quat([0, 0, 0, 1]).as_quat()  # No rotation

    # If the vectors are opposite, return a 180-degree rotation quaternion
    if np.isclose(angle, np.pi):
        # Find an arbitrary orthogonal axis
        orthogonal_axis = np.array([1, 0, 0]) if not np.allclose(v1, [1, 0, 0]) else np.array([0, 1, 0])
        axis = np.cross(v1, orthogonal_axis)
        axis = axis / np.linalg.norm(axis)
        return R.from_rotvec(np.pi * axis).as_quat()

    # Normalize the axis
    axis = axis / np.linalg.norm(axis)

    # Create quaternion from the axis-angle representation
    quaternion = R.from_rotvec(angle * axis)

    return quaternion.as_quat()
