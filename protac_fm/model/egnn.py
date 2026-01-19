"""SE(3)-Equivariant Graph Neural Network (EGNN) implementation."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum
import math


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal positional embedding for time."""
    
    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) or (N,) time values in [0, 1]
            
        Returns:
            emb: (B, dim) or (N, dim) sinusoidal embeddings
        """
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half_dim, device=t.device) / half_dim
        )
        args = t.unsqueeze(-1) * freqs
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb


class MLP(nn.Module):
    """Multi-layer perceptron."""
    
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, 
                 num_layers: int = 2, act_fn: str = 'silu'):
        super().__init__()
        
        if act_fn == 'silu':
            act = nn.SiLU()
        elif act_fn == 'relu':
            act = nn.ReLU()
        else:
            act = nn.SiLU()
        
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(act)
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act)
        
        layers.append(nn.Linear(hidden_dim, out_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EGNNLayer(nn.Module):
    """Single EGNN layer with coordinate and feature updates."""
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        act_fn: str = 'silu',
        coords_weight: float = 1.0,
        attention: bool = True,
        normalize: bool = True,
        tanh: bool = False,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.coords_weight = coords_weight
        self.attention = attention
        self.normalize = normalize
        self.tanh = tanh
        
        if act_fn == 'silu':
            self.act = nn.SiLU()
        else:
            self.act = nn.ReLU()
        
        # Edge MLP: computes edge messages
        edge_input_dim = 2 * node_dim + edge_dim + 1  # +1 for distance
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            self.act,
            nn.Linear(hidden_dim, hidden_dim),
            self.act,
        )
        
        # Node MLP: updates node features
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            self.act,
            nn.Linear(hidden_dim, node_dim),
        )
        
        # Coordinate MLP: computes coordinate updates
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            self.act,
            nn.Linear(hidden_dim, 1, bias=False),
        )
        nn.init.xavier_uniform_(self.coord_mlp[-1].weight, gain=0.001)
        
        # Attention (optional)
        if attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
    
    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_mask: torch.Tensor = None,
    ):
        """
        Args:
            h: (N, node_dim) node features
            x: (N, 3) node coordinates
            edge_index: (2, E) edge indices
            edge_attr: (E, edge_dim) edge features
            node_mask: (N,) optional mask for valid nodes
            
        Returns:
            h_out: (N, node_dim) updated node features
            x_out: (N, 3) updated coordinates
        """
        src, dst = edge_index
        
        # Compute distances
        coord_diff = x[src] - x[dst]
        dist_sq = (coord_diff ** 2).sum(dim=-1, keepdim=True)
        dist = torch.sqrt(dist_sq + 1e-8)
        
        if self.normalize:
            coord_diff = coord_diff / (dist + 1.0)
        
        # Edge features
        edge_input = torch.cat([h[src], h[dst], edge_attr, dist], dim=-1)
        edge_feat = self.edge_mlp(edge_input)
        
        # Attention
        if self.attention:
            att = self.att_mlp(edge_feat)
            edge_feat = edge_feat * att
        
        # Coordinate update
        coord_update = self.coord_mlp(edge_feat)
        if self.tanh:
            coord_update = torch.tanh(coord_update)
        coord_update = coord_diff * coord_update
        
        # Aggregate coordinate updates
        x_agg = scatter_sum(coord_update, dst, dim=0, dim_size=h.size(0))
        x_out = x + x_agg * self.coords_weight
        
        # Node feature update
        h_agg = scatter_sum(edge_feat, dst, dim=0, dim_size=h.size(0))
        h_input = torch.cat([h, h_agg], dim=-1)
        h_out = h + self.node_mlp(h_input)
        
        # Apply mask
        if node_mask is not None:
            node_mask = node_mask.unsqueeze(-1)
            h_out = h_out * node_mask
            x_out = x_out * node_mask
        
        return h_out, x_out


class EGNN(nn.Module):
    """Full EGNN model with multiple layers."""
    
    def __init__(
        self,
        in_node_dim: int,
        in_edge_dim: int,
        hidden_dim: int = 256,
        out_node_dim: int = None,
        n_layers: int = 6,
        act_fn: str = 'silu',
        attention: bool = True,
        normalize: bool = True,
        tanh: bool = False,
    ):
        super().__init__()
        
        if out_node_dim is None:
            out_node_dim = in_node_dim
        
        self.in_node_dim = in_node_dim
        self.hidden_dim = hidden_dim
        self.out_node_dim = out_node_dim
        self.n_layers = n_layers
        
        # Input embedding
        self.node_embedding = nn.Linear(in_node_dim, hidden_dim)
        self.edge_embedding = nn.Linear(in_edge_dim, hidden_dim)
        
        # EGNN layers
        self.layers = nn.ModuleList([
            EGNNLayer(
                node_dim=hidden_dim,
                edge_dim=hidden_dim,
                hidden_dim=hidden_dim,
                act_fn=act_fn,
                attention=attention,
                normalize=normalize,
                tanh=tanh,
            )
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.node_out = nn.Linear(hidden_dim, out_node_dim)
    
    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_mask: torch.Tensor = None,
    ):
        """
        Args:
            h: (N, in_node_dim) input node features
            x: (N, 3) input coordinates
            edge_index: (2, E) edge indices
            edge_attr: (E, in_edge_dim) edge features
            node_mask: (N,) optional mask
            
        Returns:
            h_out: (N, out_node_dim) output node features
            x_out: (N, 3) output coordinates
        """
        # Embed inputs
        h = self.node_embedding(h)
        edge_attr = self.edge_embedding(edge_attr)
        
        # Apply EGNN layers
        for layer in self.layers:
            h, x = layer(h, x, edge_index, edge_attr, node_mask)
        
        # Output projection
        h = self.node_out(h)
        
        return h, x
