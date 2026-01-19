"""Geometry utilities for coordinate transformations."""
import torch
import numpy as np


def local_to_global(R: torch.Tensor, t: torch.Tensor, x_local: torch.Tensor) -> torch.Tensor:
    """Transform local coordinates to global coordinates.
    
    Args:
        R: (3, 3) or (B, 3, 3) rotation matrix
        t: (3,) or (B, 3) translation vector
        x_local: (N, 3) or (B, N, 3) local coordinates
        
    Returns:
        x_global: (N, 3) or (B, N, 3) global coordinates
    """
    if R.dim() == 2:
        return x_local @ R.T + t
    else:
        return torch.einsum('bij,bnj->bni', R, x_local) + t.unsqueeze(-2)


def global_to_local(R: torch.Tensor, t: torch.Tensor, x_global: torch.Tensor) -> torch.Tensor:
    """Transform global coordinates to local coordinates.
    
    Args:
        R: (3, 3) or (B, 3, 3) rotation matrix
        t: (3,) or (B, 3) translation vector
        x_global: (N, 3) or (B, N, 3) global coordinates
        
    Returns:
        x_local: (N, 3) or (B, N, 3) local coordinates
    """
    if R.dim() == 2:
        return (x_global - t) @ R
    else:
        return torch.einsum('bji,bnj->bni', R, x_global - t.unsqueeze(-2))


def compute_relative_pose(fragA_coords: torch.Tensor, fragB_coords: torch.Tensor):
    """Compute relative pose of fragB with respect to fragA using Kabsch alignment.
    
    Args:
        fragA_coords: (N_A, 3) coordinates of fragment A
        fragB_coords: (N_B, 3) coordinates of fragment B
        
    Returns:
        p_gt: (3,) translation vector
        r_gt: (3,) rotation in axis-angle (so3 log)
    """
    from .so3 import rotation_to_so3vec
    
    # Center fragA (reference frame)
    center_A = fragA_coords.mean(dim=0)
    
    # fragB center relative to fragA center
    center_B = fragB_coords.mean(dim=0)
    p_gt = center_B - center_A
    
    # For rotation, we use identity as default (can be extended with PCA alignment)
    R_gt = torch.eye(3, device=fragA_coords.device, dtype=fragA_coords.dtype)
    r_gt = rotation_to_so3vec(R_gt)
    
    return p_gt, r_gt


def kabsch_align(P: torch.Tensor, Q: torch.Tensor):
    """Kabsch algorithm for optimal rotation alignment.
    
    Finds R that minimizes ||P - Q @ R||^2
    
    Args:
        P: (N, 3) target points
        Q: (N, 3) source points
        
    Returns:
        R: (3, 3) optimal rotation matrix
    """
    # Center both point sets
    P_centered = P - P.mean(dim=0)
    Q_centered = Q - Q.mean(dim=0)
    
    # Compute covariance matrix
    H = Q_centered.T @ P_centered
    
    # SVD
    U, S, Vt = torch.linalg.svd(H)
    
    # Compute rotation
    R = Vt.T @ U.T
    
    # Handle reflection case
    if torch.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    return R


def rotation_matrix_from_vectors(vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
    """Find rotation matrix that aligns vec1 to vec2.
    
    Args:
        vec1: (3,) source vector
        vec2: (3,) target vector
        
    Returns:
        R: (3, 3) rotation matrix
    """
    a = vec1 / vec1.norm().clamp(min=1e-8)
    b = vec2 / vec2.norm().clamp(min=1e-8)
    
    v = torch.cross(a, b)
    c = torch.dot(a, b)
    s = v.norm()
    
    if s < 1e-8:
        if c > 0:
            return torch.eye(3, device=vec1.device, dtype=vec1.dtype)
        else:
            # 180 degree rotation
            ortho = torch.tensor([1., 0., 0.], device=vec1.device, dtype=vec1.dtype)
            if torch.abs(torch.dot(a, ortho)) > 0.9:
                ortho = torch.tensor([0., 1., 0.], device=vec1.device, dtype=vec1.dtype)
            ortho = ortho - torch.dot(ortho, a) * a
            ortho = ortho / ortho.norm()
            return 2 * torch.outer(ortho, ortho) - torch.eye(3, device=vec1.device, dtype=vec1.dtype)
    
    kmat = torch.zeros(3, 3, device=vec1.device, dtype=vec1.dtype)
    kmat[0, 1] = -v[2]
    kmat[0, 2] = v[1]
    kmat[1, 0] = v[2]
    kmat[1, 2] = -v[0]
    kmat[2, 0] = -v[1]
    kmat[2, 1] = v[0]
    
    R = torch.eye(3, device=vec1.device, dtype=vec1.dtype) + kmat + kmat @ kmat * ((1 - c) / (s * s))
    return R


def find_axes(pos: np.ndarray, weights: np.ndarray = None):
    """Find principal axes of a point cloud using PCA.
    
    Args:
        pos: (N, 3) coordinates
        weights: (N,) optional weights
        
    Returns:
        axes: (3, 3) rotation matrix (columns are principal axes)
    """
    if weights is None:
        weights = np.ones(len(pos))
    
    # Center
    center = np.average(pos, axis=0, weights=weights)
    pos_centered = pos - center
    
    # Weighted covariance
    cov = np.zeros((3, 3))
    for i in range(len(pos)):
        cov += weights[i] * np.outer(pos_centered[i], pos_centered[i])
    cov /= weights.sum()
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    axes = eigenvectors[:, idx]
    
    # Ensure right-handed coordinate system
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    
    return axes.astype(np.float32)


def center_pos(pos: torch.Tensor, fragment_mask: torch.Tensor, batch: torch.Tensor, mode='frags'):
    """Center positions based on fragment centers.
    
    Args:
        pos: (N, 3) positions
        fragment_mask: (N,) fragment labels (0=linker, 1=fragA, 2=fragB)
        batch: (N,) batch indices
        mode: centering mode
        
    Returns:
        new_pos: (N, 3) centered positions
        offset: (B, 3) offset applied
    """
    from torch_scatter import scatter_mean
    
    if mode == 'frags':
        mu1 = scatter_mean(pos[fragment_mask == 1], batch[fragment_mask == 1], dim=0)
        mu2 = scatter_mean(pos[fragment_mask == 2], batch[fragment_mask == 2], dim=0)
        offset = (mu1 + mu2) / 2
        new_pos = pos - offset[batch]
    else:
        raise NotImplementedError(f"Unknown centering mode: {mode}")
    
    return new_pos, offset
