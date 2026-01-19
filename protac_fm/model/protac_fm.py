"""PROTAC Flow Matching Model."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum
from typing import Dict, Optional, Tuple

from .egnn import EGNN, MLP, SinusoidalEmbedding
from ..utils.so3 import so3vec_to_rotation, rotation_to_so3vec
from ..utils.geometry import local_to_global, center_pos


class ProTACFlowModel(nn.Module):
    """PROTAC Joint Pose + Linker Flow-Matching Model.
    
    Generates:
    - Fragment B relative pose (rotation + translation)
    - Linker atom types, bond connections, and 3D coordinates
    """
    
    def __init__(
        self,
        num_atom_types: int = 10,
        num_bond_types: int = 4,
        h_dim: int = 256,
        edge_dim: int = 8,
        cond_dim: int = 16,
        time_emb_dim: int = 64,
        n_layers: int = 6,
        num_heads: int = 4,
        N_L_max: int = 32,
    ):
        super().__init__()
        
        self.num_atom_types = num_atom_types
        self.num_bond_types = num_bond_types
        self.h_dim = h_dim
        self.edge_dim = edge_dim
        self.cond_dim = cond_dim
        self.time_emb_dim = time_emb_dim
        self.N_L_max = N_L_max
        
        # Time embedding
        self.time_emb = SinusoidalEmbedding(time_emb_dim)
        
        # Node embeddings
        self.atom_type_emb = nn.Embedding(num_atom_types, h_dim)
        self.node_kind_emb = nn.Embedding(3, h_dim)  # 0=fragA, 1=fragB, 2=linker
        self.anchor_emb = nn.Embedding(2, h_dim)  # 0=not anchor, 1=anchor
        
        # Condition embedding
        self.cond_mlp = MLP(cond_dim, h_dim, h_dim, num_layers=2)
        
        # Time projection
        self.time_proj = MLP(time_emb_dim, h_dim, h_dim, num_layers=2)
        
        # Edge embedding: 1 (same_kind) + 3 (src_kind) + 3 (dst_kind) + 1 (distance) = 8
        self.edge_input_dim = 8  # Fixed input dimension for edge features
        self.edge_mlp = MLP(self.edge_input_dim, h_dim, h_dim, num_layers=2)
        
        # Backbone EGNN
        node_input_dim = h_dim * 5  # atom + kind + anchor + time + cond
        self.backbone = EGNN(
            in_node_dim=node_input_dim,
            in_edge_dim=h_dim,
            hidden_dim=h_dim,
            out_node_dim=h_dim,
            n_layers=n_layers,
            attention=True,
        )
        
        # Output heads
        # Velocity head for linker coordinates
        self.vel_head = MLP(h_dim, 3, h_dim, num_layers=2)
        
        # Atom type head for linker
        self.atom_head = nn.Sequential(
            MLP(h_dim, h_dim, h_dim, num_layers=2),
            nn.Linear(h_dim, num_atom_types),
            nn.Softmax(dim=-1)
        )
        
        # Bond type head
        self.bond_head = nn.Sequential(
            MLP(h_dim * 2, h_dim, h_dim, num_layers=2),
            nn.Linear(h_dim, num_bond_types),
            nn.Softmax(dim=-1)
        )
        
        # Pose heads for fragment B
        # Force aggregation for rigid body motion
        self.force_head = MLP(h_dim, 3, h_dim, num_layers=2)
        
        # Translation prediction (dp)
        self.trans_head = MLP(h_dim, 3, h_dim, num_layers=2)
        
        # Rotation prediction (dr in axis-angle)
        self.rot_head = MLP(h_dim, 3, h_dim, num_layers=2)
        
        self.register_buffer('_dummy', torch.empty([0]))
    
    def forward(
        self,
        x_t: torch.Tensor,
        atom_type_t: torch.Tensor,
        node_kind: torch.Tensor,
        is_anchor: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        batch: torch.Tensor,
        linker_mask: torch.Tensor,
        fragB_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            x_t: (N, 3) noisy coordinates at time t
            atom_type_t: (N,) noisy atom types at time t
            node_kind: (N,) node kind (0=fragA, 1=fragB, 2=linker)
            is_anchor: (N,) anchor indicator
            edge_index: (2, E) edge indices
            edge_attr: (E, edge_dim) edge features
            t: (B,) time values in [0, 1]
            cond: (B, cond_dim) condition vectors
            batch: (N,) batch indices
            linker_mask: (N,) linker node mask
            fragB_mask: (N,) fragment B node mask
            
        Returns:
            Dict with:
                - vel: (N, 3) velocity for linker nodes
                - atom_logits: (N, num_atom_types) atom type logits
                - bond_logits: (E, num_bond_types) bond type logits
                - dp: (B, 3) translation update for fragB
                - dr: (B, 3) rotation update for fragB (axis-angle)
        """
        N = x_t.size(0)
        B = t.size(0)
        
        # Time embedding
        t_emb = self.time_emb(t)  # (B, time_emb_dim)
        t_node = self.time_proj(t_emb)[batch]  # (N, h_dim)
        
        # Node embeddings
        atom_emb = self.atom_type_emb(atom_type_t)  # (N, h_dim)
        kind_emb = self.node_kind_emb(node_kind)  # (N, h_dim)
        anchor_emb = self.anchor_emb(is_anchor)  # (N, h_dim)
        
        # Condition embedding (broadcast to nodes)
        cond_emb = self.cond_mlp(cond)[batch]  # (N, h_dim)
        
        # Combine node features
        h = torch.cat([atom_emb, kind_emb, anchor_emb, t_node, cond_emb], dim=-1)
        
        # Edge features
        src, dst = edge_index
        dist = (x_t[src] - x_t[dst]).norm(dim=-1, keepdim=True)
        edge_feat = torch.cat([edge_attr, dist], dim=-1)
        edge_feat = self.edge_mlp(edge_feat)
        
        # Backbone
        h_out, x_out = self.backbone(h, x_t, edge_index, edge_feat)
        
        # Velocity for linker nodes
        vel = self.vel_head(h_out)
        vel = vel * linker_mask.unsqueeze(-1).float()
        
        # Atom type logits
        atom_logits = self.atom_head(h_out)
        
        # Bond type logits (for edges involving linker)
        edge_feat_pair = torch.cat([h_out[src], h_out[dst]], dim=-1)
        bond_logits = self.bond_head(edge_feat_pair)
        
        # Pose prediction for fragB
        # Aggregate forces on fragB nodes
        fragB_h = h_out[fragB_mask]
        fragB_batch = batch[fragB_mask]
        
        # Mean pooling over fragB nodes
        fragB_agg = scatter_mean(fragB_h, fragB_batch, dim=0, dim_size=B)
        
        # Predict translation and rotation updates
        dp = self.trans_head(fragB_agg)  # (B, 3)
        dr = self.rot_head(fragB_agg)  # (B, 3)
        
        return {
            'vel': vel,
            'atom_logits': atom_logits,
            'bond_logits': bond_logits,
            'dp': dp,
            'dr': dr,
            'h_out': h_out,
            'x_out': x_out,
        }
    
    def get_loss(
        self,
        batch_data: Dict[str, torch.Tensor],
        t: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute training loss.
        
        Args:
            batch_data: Batched data dict from collate_fn
            t: Optional time values, sampled uniformly if None
            
        Returns:
            Dict of loss components
        """
        device = self._dummy.device
        B = batch_data['num_graphs']
        
        # Sample time if not provided
        if t is None:
            t = torch.rand(B, device=device)
        
        # Get ground truth data
        x_data = batch_data['x_data'].to(device)
        atom_type = batch_data['atom_type'].to(device)
        node_kind = batch_data['node_kind'].to(device)
        is_anchor = batch_data['is_anchor'].to(device)
        edge_index = batch_data['edge_index'].to(device)
        bond_type_gt = batch_data['bond_type_gt'].to(device)
        batch_idx = batch_data['batch'].to(device)
        linker_mask = batch_data['linker_mask'].to(device)
        fragB_mask = batch_data['fragB_mask'].to(device)
        cond = batch_data['cond'].to(device)
        p_gt = batch_data['p_gt'].to(device)
        r_gt = batch_data['r_gt'].to(device)
        
        # Add noise to linker coordinates (Flow Matching)
        # x_t = (1 - t) * x_data + t * x_noise
        t_node = t[batch_idx].unsqueeze(-1)
        x_noise = torch.randn_like(x_data)
        
        # Only add noise to linker nodes
        x_t = x_data.clone()
        x_t[linker_mask] = (1 - t_node[linker_mask]) * x_data[linker_mask] + t_node[linker_mask] * x_noise[linker_mask]
        
        # Target velocity: u_t = x_data - x_t (for linker)
        u_t = x_data - x_t
        
        # Mask atom types for linker based on time
        atom_type_t = atom_type.clone()
        mask_prob = t[batch_idx]
        mask_rand = torch.rand_like(mask_prob)
        should_mask = (mask_rand < mask_prob) & linker_mask
        atom_type_t[should_mask] = 0  # MASK token
        
        # Compute edge attributes
        edge_attr = self._compute_edge_attr(x_t, edge_index, node_kind)
        
        # Forward pass
        outputs = self.forward(
            x_t=x_t,
            atom_type_t=atom_type_t,
            node_kind=node_kind,
            is_anchor=is_anchor,
            edge_index=edge_index,
            edge_attr=edge_attr,
            t=t,
            cond=cond,
            batch=batch_idx,
            linker_mask=linker_mask,
            fragB_mask=fragB_mask,
        )
        
        # Coordinate loss (MSE on velocity for linker)
        vel_pred = outputs['vel']
        loss_coord = F.mse_loss(vel_pred[linker_mask], u_t[linker_mask], reduction='mean')
        
        # Pose loss
        dp_pred = outputs['dp']
        dr_pred = outputs['dr']
        
        # Current pose (simplified: use mean position)
        fragB_pos = x_t[fragB_mask]
        fragB_batch = batch_idx[fragB_mask]
        p_current = scatter_mean(fragB_pos, fragB_batch, dim=0, dim_size=B)
        
        # Target: push towards ground truth
        dp_target = p_gt - p_current
        dr_target = r_gt  # Simplified: direct prediction
        
        loss_pose = F.mse_loss(dp_pred, dp_target) + F.mse_loss(dr_pred, dr_target)
        
        # Atom type loss (cross-entropy for linker)
        atom_logits = outputs['atom_logits']
        loss_atom = F.cross_entropy(
            atom_logits[linker_mask],
            atom_type[linker_mask],
            reduction='mean'
        )
        
        # Bond type loss
        bond_logits = outputs['bond_logits']
        loss_bond = F.cross_entropy(bond_logits, bond_type_gt, reduction='mean')
        
        return {
            'coord': loss_coord,
            'pose': loss_pose,
            'atom': loss_atom,
            'bond': loss_bond,
        }
    
    def _compute_edge_attr(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_kind: torch.Tensor,
    ) -> torch.Tensor:
        """Compute edge attributes."""
        src, dst = edge_index
        
        # Same fragment indicator
        same_kind = (node_kind[src] == node_kind[dst]).float().unsqueeze(-1)
        
        # Node kind one-hot
        src_kind = F.one_hot(node_kind[src], num_classes=3).float()
        dst_kind = F.one_hot(node_kind[dst], num_classes=3).float()
        
        edge_attr = torch.cat([same_kind, src_kind, dst_kind], dim=-1)
        
        return edge_attr
    
    @torch.no_grad()
    def sample(
        self,
        batch_data: Dict[str, torch.Tensor],
        n_steps: int = 80,
        dt: float = 0.0125,
    ) -> Dict[str, torch.Tensor]:
        """Sample from the model using ODE integration.
        
        Args:
            batch_data: Batched data dict (fragments only, linker to be generated)
            n_steps: Number of ODE steps
            dt: Time step size
            
        Returns:
            Dict with generated coordinates, atom types, and bond types
        """
        device = self._dummy.device
        B = batch_data['num_graphs']
        
        # Get data
        x_data = batch_data['x_data'].to(device)
        atom_type = batch_data['atom_type'].to(device)
        node_kind = batch_data['node_kind'].to(device)
        is_anchor = batch_data['is_anchor'].to(device)
        edge_index = batch_data['edge_index'].to(device)
        batch_idx = batch_data['batch'].to(device)
        linker_mask = batch_data['linker_mask'].to(device)
        fragB_mask = batch_data['fragB_mask'].to(device)
        cond = batch_data['cond'].to(device)
        
        # Initialize linker coordinates from noise
        x_t = x_data.clone()
        x_t[linker_mask] = torch.randn_like(x_t[linker_mask])
        
        # Initialize atom types as MASK
        atom_type_t = atom_type.clone()
        atom_type_t[linker_mask] = 0  # MASK
        
        # Initialize fragB pose randomly
        # (In practice, would initialize from prior)
        
        # ODE integration from t=1 to t=0
        for step in range(n_steps):
            t_val = 1.0 - step * dt
            t = torch.full((B,), t_val, device=device)
            
            # Compute edge attributes
            edge_attr = self._compute_edge_attr(x_t, edge_index, node_kind)
            
            # Forward pass
            outputs = self.forward(
                x_t=x_t,
                atom_type_t=atom_type_t,
                node_kind=node_kind,
                is_anchor=is_anchor,
                edge_index=edge_index,
                edge_attr=edge_attr,
                t=t,
                cond=cond,
                batch=batch_idx,
                linker_mask=linker_mask,
                fragB_mask=fragB_mask,
            )
            
            # Update coordinates (Euler step)
            vel = outputs['vel']
            x_t = x_t + vel * dt
            
            # Unmask atoms based on schedule
            unmask_prob = dt / (t_val + 1e-8)
            unmask_rand = torch.rand(linker_mask.sum(), device=device)
            linker_indices = linker_mask.nonzero(as_tuple=True)[0]
            
            for i, idx in enumerate(linker_indices):
                if atom_type_t[idx] == 0 and unmask_rand[i] < unmask_prob:
                    # Sample atom type
                    probs = outputs['atom_logits'][idx]
                    atom_type_t[idx] = torch.multinomial(probs, 1).item()
        
        # Final bond prediction
        edge_attr = self._compute_edge_attr(x_t, edge_index, node_kind)
        outputs = self.forward(
            x_t=x_t,
            atom_type_t=atom_type_t,
            node_kind=node_kind,
            is_anchor=is_anchor,
            edge_index=edge_index,
            edge_attr=edge_attr,
            t=torch.zeros(B, device=device),
            cond=cond,
            batch=batch_idx,
            linker_mask=linker_mask,
            fragB_mask=fragB_mask,
        )
        
        # Sample bond types
        bond_types = torch.argmax(outputs['bond_logits'], dim=-1)
        
        return {
            'x_final': x_t,
            'atom_type_final': atom_type_t,
            'bond_type_final': bond_types,
        }
