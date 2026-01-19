"""Mask scheduling for discrete denoising."""
import torch
import torch.nn as nn
from typing import Literal


class MaskSchedule:
    """Schedule for masking discrete tokens during training."""
    
    def __init__(self, schedule_type: Literal['linear', 'cosine', 'sqrt'] = 'linear'):
        """Initialize mask schedule.
        
        Args:
            schedule_type: Type of schedule
                - 'linear': p(mask) = t
                - 'cosine': p(mask) = 1 - cos(pi/2 * (1-t))
                - 'sqrt': p(mask) = sqrt(t)
        """
        self.schedule_type = schedule_type
    
    def get_mask_prob(self, t: torch.Tensor) -> torch.Tensor:
        """Get masking probability at time t.
        
        Args:
            t: Time values in [0, 1], where t=0 is clean and t=1 is fully masked
            
        Returns:
            Probability of masking at time t
        """
        if self.schedule_type == 'linear':
            return t
        elif self.schedule_type == 'cosine':
            return 1 - torch.cos(torch.pi / 2 * (1 - t))
        elif self.schedule_type == 'sqrt':
            return torch.sqrt(t)
        else:
            return t
    
    def mask_tokens(
        self,
        tokens: torch.Tensor,
        t: torch.Tensor,
        mask_token: int = 0,
    ) -> torch.Tensor:
        """Apply masking to tokens based on time.
        
        Args:
            tokens: (N,) token indices
            t: (N,) time values for each token
            mask_token: Index of mask token
            
        Returns:
            Masked tokens
        """
        mask_prob = self.get_mask_prob(t)
        mask_rand = torch.rand_like(mask_prob)
        should_mask = mask_rand < mask_prob
        
        masked_tokens = tokens.clone()
        masked_tokens[should_mask] = mask_token
        
        return masked_tokens
    
    def get_unmask_schedule(self, n_steps: int) -> torch.Tensor:
        """Get unmask schedule for sampling.
        
        Args:
            n_steps: Number of sampling steps
            
        Returns:
            (n_steps,) tensor of unmask probabilities per step
        """
        t_values = torch.linspace(1, 0, n_steps + 1)
        mask_probs = self.get_mask_prob(t_values)
        
        # Probability of unmasking at each step
        unmask_probs = mask_probs[:-1] - mask_probs[1:]
        unmask_probs = unmask_probs / mask_probs[:-1].clamp(min=1e-8)
        
        return unmask_probs


class DiscreteDenoiser:
    """Discrete denoising for atom and bond types."""
    
    def __init__(
        self,
        num_classes: int,
        mask_token: int = 0,
        schedule: MaskSchedule = None,
    ):
        """Initialize discrete denoiser.
        
        Args:
            num_classes: Number of token classes (including mask)
            mask_token: Index of mask token
            schedule: Mask schedule to use
        """
        self.num_classes = num_classes
        self.mask_token = mask_token
        self.schedule = schedule or MaskSchedule('linear')
    
    def add_noise(
        self,
        tokens: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise (masking) to tokens.
        
        Args:
            tokens: (N,) clean token indices
            t: (N,) or scalar time values
            
        Returns:
            Noisy (masked) tokens
        """
        if t.dim() == 0:
            t = t.expand(tokens.size(0))
        
        return self.schedule.mask_tokens(tokens, t, self.mask_token)
    
    def posterior(
        self,
        tokens_t: torch.Tensor,
        tokens_pred: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute posterior distribution for denoising.
        
        Args:
            tokens_t: (N,) current noisy tokens
            tokens_pred: (N, num_classes) predicted probabilities
            t: (N,) time values
            
        Returns:
            (N, num_classes) posterior probabilities
        """
        # For masked tokens, use predicted distribution
        # For unmasked tokens, keep current token
        is_masked = (tokens_t == self.mask_token)
        
        posterior = torch.zeros_like(tokens_pred)
        
        # Masked: use prediction
        posterior[is_masked] = tokens_pred[is_masked]
        
        # Unmasked: one-hot of current token
        unmasked_indices = (~is_masked).nonzero(as_tuple=True)[0]
        posterior[unmasked_indices] = torch.nn.functional.one_hot(
            tokens_t[unmasked_indices], num_classes=self.num_classes
        ).float()
        
        return posterior
    
    def sample_step(
        self,
        tokens_t: torch.Tensor,
        logits: torch.Tensor,
        t_current: float,
        t_next: float,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Sample one denoising step.
        
        Args:
            tokens_t: (N,) current tokens
            logits: (N, num_classes) predicted logits
            t_current: Current time
            t_next: Next time (closer to 0)
            temperature: Sampling temperature
            
        Returns:
            Updated tokens
        """
        # Probability of unmasking
        p_unmask = (self.schedule.get_mask_prob(torch.tensor(t_current)) - 
                    self.schedule.get_mask_prob(torch.tensor(t_next)))
        p_unmask = p_unmask / self.schedule.get_mask_prob(torch.tensor(t_current)).clamp(min=1e-8)
        
        # Sample which tokens to unmask
        is_masked = (tokens_t == self.mask_token)
        unmask_rand = torch.rand(tokens_t.size(0), device=tokens_t.device)
        should_unmask = is_masked & (unmask_rand < p_unmask)
        
        # Sample new tokens for unmasked positions
        probs = torch.softmax(logits / temperature, dim=-1)
        sampled = torch.multinomial(probs, 1).squeeze(-1)
        
        tokens_next = tokens_t.clone()
        tokens_next[should_unmask] = sampled[should_unmask]
        
        return tokens_next
