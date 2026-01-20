"""PROTAC Flow Matching Model compatible with LinkerNet data format."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum
from typing import Dict, Optional, Tuple
import numpy as np

from .egnn import EGNN, MLP, SinusoidalEmbedding


class ProTACFlowModel(nn.Module):
    """PROTAC Joint Pose + Linker Flow-Matching Model.
    
    Compatible with LinkerNet data format.
    """
    
    def __init__(
        self,
        num_atom_types: int = 14,
        num_bond_types: int = 5,
        h_dim: int = 256,
        edge_dim: int = 64,
        cond_dim: int = 16,
        time_emb_dim: int = 64,
        n_layers: int = 6,
        num_heads: int = 4,
        N_L_max: int = 50,
        num_steps: int = 500,
    ):
        super().__init__()
        
        self.num_atom_types = num_atom_types
        self.num_bond_types = num_bond_types
        self.h_dim = h_dim
        self.edge_dim = edge_dim
        self.cond_dim = cond_dim
        self.time_emb_dim = time_emb_dim
        self.N_L_max = N_L_max
        self.num_steps = num_steps
        self.train_bond = True
        
        # Time embedding
        self.time_emb = SinusoidalEmbedding(time_emb_dim)
        
        # Node embeddings
        self.atom_type_emb = nn.Embedding(num_atom_types + 1, h_dim)  # +1 for mask
        self.fragment_emb = nn.Embedding(3, h_dim)  # 0=linker, 1=frag1, 2=frag2
        
        # Atom feature projection (dimension depends on known_anchor setting)
        self.atom_feat_proj = nn.Linear(2, h_dim)  # linker_flag(2), or 4 if known_anchor
        
        # Time projection
        self.time_proj = MLP(time_emb_dim, h_dim, h_dim, num_layers=2)
        
        # Edge embedding
        self.edge_feat_proj = nn.Linear(4, h_dim)  # edge type one-hot
        
        # Backbone EGNN
        node_input_dim = h_dim * 4  # atom + fragment + atom_feat + time
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
        
        # Atom type head
        self.atom_head = nn.Sequential(
            MLP(h_dim, h_dim, h_dim, num_layers=2),
            nn.Linear(h_dim, num_atom_types),
        )
        
        # Bond type head
        self.bond_head = nn.Sequential(
            MLP(h_dim * 2, h_dim, h_dim, num_layers=2),
            nn.Linear(h_dim, num_bond_types),
        )
        
        # Pose heads for fragments
        self.trans_head = MLP(h_dim, 3, h_dim, num_layers=2)
        self.rot_head = MLP(h_dim, 3, h_dim, num_layers=2)
        
        self.register_buffer('_dummy', torch.empty([0]))
    
    def forward(
        self,
        batch,
        t: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            batch: PyG batch object with LinkerNet format
            t: (B,) time values in [0, num_steps-1]
            
        Returns:
            Dict with predictions
        """
        device = self._dummy.device
        batch_idx = batch.batch
        B = int(batch_idx.max().item()) + 1 if len(batch_idx) > 0 else 1
        
        # Get data from batch
        pos = batch.pos
        atom_type = batch.atom_type
        fragment_mask = batch.fragment_mask
        edge_index = batch.edge_index
        edge_feat = batch.edge_feat
        atom_feat = batch.atom_feat
        batch_idx = batch.batch
        
        linker_mask = (fragment_mask == 0)
        frag1_mask = (fragment_mask == 1)
        frag2_mask = (fragment_mask == 2)
        
        # Time embedding
        if t is None:
            t = torch.zeros(B, device=device)
        t_normalized = t.float() / self.num_steps
        t_emb = self.time_emb(t_normalized)
        t_node = self.time_proj(t_emb)[batch_idx]
        
        # Node embeddings
        atom_emb = self.atom_type_emb(atom_type)
        frag_emb = self.fragment_emb(fragment_mask)
        feat_emb = self.atom_feat_proj(atom_feat)
        
        # Combine node features
        h = torch.cat([atom_emb, frag_emb, feat_emb, t_node], dim=-1)
        
        # Edge features
        edge_emb = self.edge_feat_proj(edge_feat)
        
        # Backbone
        h_out, x_out = self.backbone(h, pos, edge_index, edge_emb)
        
        # Velocity for linker nodes
        vel = self.vel_head(h_out)
        vel = vel * linker_mask.unsqueeze(-1).float()
        
        # Atom type logits
        atom_logits = self.atom_head(h_out)
        
        # Bond type logits
        src, dst = edge_index
        edge_feat_pair = torch.cat([h_out[src], h_out[dst]], dim=-1)
        bond_logits = self.bond_head(edge_feat_pair)
        
        # Pose prediction for fragments
        frag1_h = h_out[frag1_mask]
        frag1_batch = batch_idx[frag1_mask]
        frag1_agg = scatter_mean(frag1_h, frag1_batch, dim=0, dim_size=B)
        
        frag2_h = h_out[frag2_mask]
        frag2_batch = batch_idx[frag2_mask]
        frag2_agg = scatter_mean(frag2_h, frag2_batch, dim=0, dim_size=B)
        
        dp1 = self.trans_head(frag1_agg)
        dr1 = self.rot_head(frag1_agg)
        dp2 = self.trans_head(frag2_agg)
        dr2 = self.rot_head(frag2_agg)
        
        return {
            'vel': vel,
            'atom_logits': atom_logits,
            'bond_logits': bond_logits,
            'dp1': dp1, 'dr1': dr1,
            'dp2': dp2, 'dr2': dr2,
            'h_out': h_out,
            'x_out': x_out,
        }
    
    def get_loss(
        self,
        batch,
        t: torch.Tensor = None,
        pos_noise_std: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """Compute training loss.
        
        Args:
            batch: PyG batch object
            t: Optional time values
            pos_noise_std: Position noise std for data augmentation
            
        Returns:
            Dict of loss components
        """
        device = self._dummy.device
        batch = batch.to(device)
        B = batch.num_graphs
        
        # Sample time if not provided
        if t is None:
            t = torch.randint(0, self.num_steps, (B,), device=device)
        else:
            t = t.to(device)
        
        t_normalized = t.float() / self.num_steps
        
        # Get data
        pos = batch.pos.clone()
        atom_type = batch.atom_type
        fragment_mask = batch.fragment_mask
        edge_index = batch.edge_index
        edge_type = batch.edge_type
        batch_idx = batch.batch
        
        linker_mask = (fragment_mask == 0)
        frag_mask = (fragment_mask > 0)
        
        # Add position noise for augmentation
        if pos_noise_std > 0:
            pos = pos + torch.randn_like(pos) * pos_noise_std
        
        # Flow matching: add noise to linker positions
        t_node = t_normalized[batch_idx].unsqueeze(-1)
        noise = torch.randn_like(pos)
        
        # Noisy positions for linker
        pos_noisy = pos.clone()
        pos_noisy[linker_mask] = (1 - t_node[linker_mask]) * pos[linker_mask] + t_node[linker_mask] * noise[linker_mask]
        
        # Target velocity
        vel_target = pos - pos_noisy
        
        # Mask atom types for linker
        atom_type_noisy = atom_type.clone()
        mask_prob = t_normalized[batch_idx]
        mask_rand = torch.rand(len(atom_type), device=device)
        should_mask = (mask_rand < mask_prob) & linker_mask
        atom_type_noisy[should_mask] = self.num_atom_types  # Mask token
        
        # Create noisy batch
        batch_noisy = batch.clone()
        batch_noisy.pos = pos_noisy
        batch_noisy.atom_type = atom_type_noisy
        
        # Forward pass
        outputs = self.forward(batch_noisy, t)
        
        # Losses
        losses = {}
        
        # Linker position loss (velocity matching)
        vel_pred = outputs['vel']
        losses['link_pos'] = F.mse_loss(vel_pred[linker_mask], vel_target[linker_mask])
        
        # Atom type loss
        atom_logits = outputs['atom_logits']
        losses['link_cls'] = F.cross_entropy(atom_logits[linker_mask], atom_type[linker_mask])
        
        # Bond type loss
        bond_logits = outputs['bond_logits']
        linker_bond_mask = batch.linker_bond_mask
        if linker_bond_mask.sum() > 0:
            bond_logits_masked = bond_logits[linker_bond_mask]
            bond_target = edge_type[linker_bond_mask]

            class_counts = torch.bincount(bond_target, minlength=self.num_bond_types).float()
            inv = class_counts.sum() / (class_counts + 1e-6)
            inv = inv / inv.mean().clamp_min(1e-6)
            inv = inv.clamp(min=0.1, max=20.0)
            losses['link_bond'] = F.cross_entropy(bond_logits_masked, bond_target, weight=inv)

            with torch.no_grad():
                pred_bond = bond_logits_masked.argmax(dim=-1)
                src, dst = edge_index
                cross_edge = ((fragment_mask[src] == 0) & (fragment_mask[dst] > 0)) | ((fragment_mask[dst] == 0) & (fragment_mask[src] > 0))
                cross_edge_masked = cross_edge[linker_bond_mask]

                cross_gt_pos = (bond_target > 0) & cross_edge_masked
                if cross_gt_pos.any():
                    cross_pred_pos = (pred_bond > 0) & cross_edge_masked
                    losses['cross_bond_hit'] = (cross_pred_pos[cross_gt_pos].float().mean())

                    frag1_touch = ((fragment_mask[src] == 1) | (fragment_mask[dst] == 1))
                    frag2_touch = ((fragment_mask[src] == 2) | (fragment_mask[dst] == 2))
                    frag1_touch_m = frag1_touch[linker_bond_mask] & cross_edge_masked
                    frag2_touch_m = frag2_touch[linker_bond_mask] & cross_edge_masked

                    gt_f1 = cross_gt_pos & frag1_touch_m
                    gt_f2 = cross_gt_pos & frag2_touch_m
                    if gt_f1.any():
                        losses['cross_bond_hit_f1'] = (cross_pred_pos[gt_f1].float().mean())
                    else:
                        losses['cross_bond_hit_f1'] = torch.tensor(0.0, device=device)
                    if gt_f2.any():
                        losses['cross_bond_hit_f2'] = (cross_pred_pos[gt_f2].float().mean())
                    else:
                        losses['cross_bond_hit_f2'] = torch.tensor(0.0, device=device)
                else:
                    losses['cross_bond_hit'] = torch.tensor(0.0, device=device)
                    losses['cross_bond_hit_f1'] = torch.tensor(0.0, device=device)
                    losses['cross_bond_hit_f2'] = torch.tensor(0.0, device=device)
        else:
            losses['link_bond'] = torch.tensor(0.0, device=device)
            losses['cross_bond_hit'] = torch.tensor(0.0, device=device)
            losses['cross_bond_hit_f1'] = torch.tensor(0.0, device=device)
            losses['cross_bond_hit_f2'] = torch.tensor(0.0, device=device)
        
        # Fragment pose losses
        frag1_mask = (fragment_mask == 1)
        frag2_mask = (fragment_mask == 2)
        
        # Fragment position loss
        frag1_pos = pos[frag1_mask]
        frag1_batch = batch_idx[frag1_mask]
        frag1_center = scatter_mean(frag1_pos, frag1_batch, dim=0, dim_size=B)
        
        frag2_pos = pos[frag2_mask]
        frag2_batch = batch_idx[frag2_mask]
        frag2_center = scatter_mean(frag2_pos, frag2_batch, dim=0, dim_size=B)
        
        # Target: maintain fragment positions
        dp1_target = torch.zeros_like(outputs['dp1'])
        dp2_target = torch.zeros_like(outputs['dp2'])
        
        losses['frag_pos'] = F.mse_loss(outputs['dp1'], dp1_target) + F.mse_loss(outputs['dp2'], dp2_target)
        
        # Fragment rotation loss (simplified)
        dr1_target = torch.zeros_like(outputs['dr1'])
        dr2_target = torch.zeros_like(outputs['dr2'])
        losses['frag_rot'] = F.mse_loss(outputs['dr1'], dr1_target) + F.mse_loss(outputs['dr2'], dr2_target)
        
        return losses
    
    @torch.no_grad()
    def sample(
        self,
        batch,
        n_steps: int = 100,
    ) -> Dict[str, torch.Tensor]:
        """Sample from the model.
        
        Args:
            batch: PyG batch object (fragments, linker to be generated)
            n_steps: Number of sampling steps
            
        Returns:
            Dict with generated results
        """
        device = self._dummy.device
        batch = batch.to(device)
        B = batch.num_graphs
        
        # Get data
        pos = batch.pos.clone()
        atom_type = batch.atom_type.clone()
        fragment_mask = batch.fragment_mask
        batch_idx = batch.batch
        
        linker_mask = (fragment_mask == 0)
        
        # Initialize linker from noise
        pos[linker_mask] = torch.randn_like(pos[linker_mask])
        
        # Initialize atom types as mask
        atom_type[linker_mask] = self.num_atom_types
        
        # Sampling loop
        dt = 1.0 / n_steps
        
        for step in range(n_steps):
            t_val = 1.0 - step * dt
            t = torch.full((B,), int(t_val * self.num_steps), device=device)
            
            # Update batch
            batch_t = batch.clone()
            batch_t.pos = pos
            batch_t.atom_type = atom_type
            
            # Forward pass
            outputs = self.forward(batch_t, t)
            
            # Update positions
            vel = outputs['vel']
            pos = pos + vel * dt
            
            # Unmask atoms
            unmask_prob = dt / (t_val + 1e-8)
            linker_indices = linker_mask.nonzero(as_tuple=True)[0]
            
            for idx in linker_indices:
                if atom_type[idx] == self.num_atom_types:
                    if torch.rand(1, device=device) < unmask_prob:
                        probs = F.softmax(outputs['atom_logits'][idx], dim=-1)
                        atom_type[idx] = torch.multinomial(probs, 1).item()
        
        # Final bond prediction
        batch_final = batch.clone()
        batch_final.pos = pos
        batch_final.atom_type = atom_type
        
        outputs = self.forward(batch_final, torch.zeros(B, device=device, dtype=torch.long))
        bond_types = torch.argmax(outputs['bond_logits'], dim=-1)
        
        # Reshape outputs per sample
        final_x = []
        final_c = []
        final_bond = []
        
        ptr = batch.ptr if hasattr(batch, 'ptr') else None
        if ptr is None:
            # Compute ptr from batch
            ptr = [0]
            for i in range(B):
                ptr.append(ptr[-1] + (batch_idx == i).sum().item())
            ptr = torch.tensor(ptr, device=device)
        
        for i in range(B):
            start, end = ptr[i].item(), ptr[i + 1].item()
            final_x.append(pos[start:end])
            final_c.append(atom_type[start:end])
        
        # Handle bond types per sample
        edge_ptr = [0]
        for i in range(B):
            n_edges = ((batch_idx[batch.edge_index[0]] == i)).sum().item()
            edge_ptr.append(edge_ptr[-1] + n_edges)
        
        for i in range(B):
            start, end = edge_ptr[i], edge_ptr[i + 1]
            final_bond.append(bond_types[start:end])
        
        return {
            'x_final': final_x,
            'atom_type_final': final_c,
            'bond_type_final': final_bond,
        }
