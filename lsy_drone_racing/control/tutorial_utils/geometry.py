"""Geometry helpers for the controls tutorial."""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation as R


def force_to_action(f_des: np.ndarray, yaw_des: float, quat: np.ndarray) -> np.ndarray:
    """Convert a desired world-frame force into the env's attitude action.

    A quadrotor can only push along its body z-axis, so to produce ``f_des`` it must tilt its
    z-axis onto the direction of ``f_des`` and set the thrust magnitude.

    Args:
        f_des: Desired force on the drone in the world frame (N), shape (3,).
        yaw_des: Desired heading (rad). Yaw is free once the thrust direction is fixed.
        quat: Current attitude quaternion, xyzw (scipy convention), shape (4,).

    Returns:
        Action ``[roll, pitch, yaw, thrust]`` as float32. Angles in rad, thrust in N.
    """
    f_des = np.asarray(f_des, dtype=float)
    norm = np.linalg.norm(f_des)
    if norm < 1e-6:  # No meaningful force requested: stay level, cut thrust.
        return np.array([0.0, 0.0, yaw_des, 0.0], dtype=np.float32)

    # Thrust: the part of f_des along the drone's CURRENT body z-axis.
    z_axis = R.from_quat(quat).as_matrix()[:, 2]
    thrust = max(float(f_des.dot(z_axis)), 0.0)  # Motors can't push downward.

    # Desired body z-axis points along the requested force.
    z_des = f_des / norm

    # Build a right-handed frame with z along z_des and x as close as possible to the heading.
    x_heading = np.array([math.cos(yaw_des), math.sin(yaw_des), 0.0])
    y_des = np.cross(z_des, x_heading)
    y_norm = np.linalg.norm(y_des)
    if y_norm < 1e-6:  # Force parallel to the heading (degenerate); use a perpendicular instead.
        x_heading = np.array([-math.sin(yaw_des), math.cos(yaw_des), 0.0])
        y_des = np.cross(z_des, x_heading)
        y_norm = np.linalg.norm(y_des)
    y_des /= y_norm
    x_des = np.cross(y_des, z_des)

    rpy = R.from_matrix(np.column_stack([x_des, y_des, z_des])).as_euler("xyz")
    return np.concatenate([rpy, [thrust]]).astype(np.float32)