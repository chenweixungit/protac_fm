"""SO(3) utilities for rotation representations."""
import torch
import numpy as np


def so3vec_to_rotation(v: torch.Tensor) -> torch.Tensor:
    """Convert SO(3) vector (axis-angle) to rotation matrix.
    
    Args:
        v: (*, 3) axis-angle representation
        
    Returns:
        R: (*, 3, 3) rotation matrix
    """
    theta = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    k = v / theta  # unit axis
    
    K = torch.zeros(*v.shape[:-1], 3, 3, device=v.device, dtype=v.dtype)
    K[..., 0, 1] = -k[..., 2]
    K[..., 0, 2] = k[..., 1]
    K[..., 1, 0] = k[..., 2]
    K[..., 1, 2] = -k[..., 0]
    K[..., 2, 0] = -k[..., 1]
    K[..., 2, 1] = k[..., 0]
    
    I = torch.eye(3, device=v.device, dtype=v.dtype).expand(*v.shape[:-1], 3, 3)
    theta = theta.unsqueeze(-1)
    
    R = I + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)
    return R


def rotation_to_so3vec(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to SO(3) vector (axis-angle).
    
    Args:
        R: (*, 3, 3) rotation matrix
        
    Returns:
        v: (*, 3) axis-angle representation
    """
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    theta = torch.acos(torch.clamp((trace - 1) / 2, -1 + 1e-7, 1 - 1e-7))
    
    # Handle small angles
    small_angle = theta.abs() < 1e-6
    
    # General case
    k = torch.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1]
    ], dim=-1)
    
    sin_theta = torch.sin(theta).unsqueeze(-1).clamp(min=1e-8)
    k = k / (2 * sin_theta)
    v = k * theta.unsqueeze(-1)
    
    # Small angle approximation
    v_small = torch.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1]
    ], dim=-1) / 2
    
    v = torch.where(small_angle.unsqueeze(-1), v_small, v)
    return v


def random_uniform_so3(shape, device='cpu'):
    """Sample random rotation in SO(3) uniformly.
    
    Args:
        shape: batch shape
        device: torch device
        
    Returns:
        v: (*shape, 3) axis-angle representation
    """
    if isinstance(shape, int):
        shape = [shape]
    
    # Sample quaternion uniformly
    u = torch.rand(*shape, 3, device=device)
    
    sqrt1_u0 = torch.sqrt(1 - u[..., 0])
    sqrt_u0 = torch.sqrt(u[..., 0])
    
    q = torch.stack([
        sqrt1_u0 * torch.sin(2 * np.pi * u[..., 1]),
        sqrt1_u0 * torch.cos(2 * np.pi * u[..., 1]),
        sqrt_u0 * torch.sin(2 * np.pi * u[..., 2]),
        sqrt_u0 * torch.cos(2 * np.pi * u[..., 2])
    ], dim=-1)
    
    # Convert quaternion to rotation matrix then to axis-angle
    R = quaternion_to_rotation(q)
    return rotation_to_so3vec(R)


def quaternion_to_rotation(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion to rotation matrix.
    
    Args:
        q: (*, 4) quaternion [x, y, z, w]
        
    Returns:
        R: (*, 3, 3) rotation matrix
    """
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
    R = torch.stack([
        1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w,
        2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w,
        2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)
    
    return R


def interpolate_so3(v0: torch.Tensor, v1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Interpolate between two SO(3) vectors using SLERP.
    
    Args:
        v0: (*, 3) start axis-angle
        v1: (*, 3) end axis-angle
        t: (*,) or scalar interpolation parameter in [0, 1]
        
    Returns:
        v_t: (*, 3) interpolated axis-angle
    """
    R0 = so3vec_to_rotation(v0)
    R1 = so3vec_to_rotation(v1)
    
    # R_t = R0 @ exp(t * log(R0^T @ R1))
    R_rel = R0.transpose(-1, -2) @ R1
    v_rel = rotation_to_so3vec(R_rel)
    
    if t.dim() == 0:
        t = t.unsqueeze(-1)
    while t.dim() < v_rel.dim():
        t = t.unsqueeze(-1)
    
    v_t_rel = t * v_rel
    R_t = R0 @ so3vec_to_rotation(v_t_rel)
    
    return rotation_to_so3vec(R_t)
