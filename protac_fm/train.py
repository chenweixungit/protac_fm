"""Training script for PROTAC Flow Matching model."""
import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

from .model import ProTACFlowModel
from .data import ProtacDataModule
from .utils.seed import seed_all


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_optimizer(config: dict, model: nn.Module) -> torch.optim.Optimizer:
    """Create optimizer from config."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=config['train']['lr'],
        weight_decay=config['train'].get('weight_decay', 0.0)
    )


def get_scheduler(config: dict, optimizer: torch.optim.Optimizer):
    """Create learning rate scheduler from config."""
    scheduler_type = config['train'].get('scheduler', 'plateau')
    
    if scheduler_type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
    elif scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['train']['max_steps']
        )
    else:
        return None


def sum_weighted_losses(loss_dict: dict, weights: dict) -> torch.Tensor:
    """Sum losses with weights."""
    total = 0.0
    for key, loss in loss_dict.items():
        weight = weights.get(f'w_{key}', 1.0)
        total = total + weight * loss
    return total


class Trainer:
    """Training manager for PROTAC Flow Matching model."""
    
    def __init__(
        self,
        config: dict,
        model: ProTACFlowModel,
        data_module: ProtacDataModule,
        device: str = 'cuda',
        log_dir: str = './logs',
    ):
        self.config = config
        self.model = model.to(device)
        self.data_module = data_module
        self.device = device
        
        # Setup logging
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.log_dir / 'checkpoints'
        self.ckpt_dir.mkdir(exist_ok=True)
        
        self.writer = SummaryWriter(self.log_dir / 'tensorboard')
        
        # Optimizer and scheduler
        self.optimizer = get_optimizer(config, model)
        self.scheduler = get_scheduler(config, self.optimizer)
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def train_step(self, batch: dict) -> dict:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        # Forward pass
        loss_dict = self.model.get_loss(batch)
        
        # Compute total loss
        weights = self.config['train']
        total_loss = sum_weighted_losses(loss_dict, weights)
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        max_grad_norm = self.config['train'].get('max_grad_norm', 10.0)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        
        # Optimizer step
        self.optimizer.step()
        
        loss_dict['total'] = total_loss
        return {k: v.item() for k, v in loss_dict.items()}
    
    @torch.no_grad()
    def validate(self, val_loader) -> dict:
        """Run validation."""
        self.model.eval()
        
        total_losses = {}
        num_batches = 0
        
        for batch in tqdm(val_loader, desc='Validating'):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            loss_dict = self.model.get_loss(batch)
            weights = self.config['train']
            total_loss = sum_weighted_losses(loss_dict, weights)
            loss_dict['total'] = total_loss
            
            for k, v in loss_dict.items():
                if k not in total_losses:
                    total_losses[k] = 0.0
                total_losses[k] += v.item()
            
            num_batches += 1
        
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        return avg_losses
    
    def save_checkpoint(self, name: str = 'latest'):
        """Save model checkpoint."""
        ckpt = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }
        torch.save(ckpt, self.ckpt_dir / f'{name}.pt')
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        if self.scheduler and ckpt['scheduler']:
            self.scheduler.load_state_dict(ckpt['scheduler'])
        self.global_step = ckpt['global_step']
        self.best_val_loss = ckpt['best_val_loss']
    
    def train(self):
        """Main training loop."""
        config = self.config['train']
        max_steps = config['max_steps']
        val_freq = config['val_freq']
        log_freq = config.get('log_freq', 100)
        
        # Setup data
        self.data_module.setup()
        train_loader = self.data_module.train_dataloader()
        val_loader = self.data_module.val_dataloader()
        
        # Training loop
        train_iter = iter(train_loader)
        pbar = tqdm(total=max_steps, desc='Training')
        pbar.update(self.global_step)
        
        while self.global_step < max_steps:
            # Get batch
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            
            # Train step
            losses = self.train_step(batch)
            self.global_step += 1
            pbar.update(1)
            
            # Logging
            if self.global_step % log_freq == 0:
                for k, v in losses.items():
                    self.writer.add_scalar(f'train/{k}', v, self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
                
                pbar.set_postfix({k: f'{v:.4f}' for k, v in losses.items()})
            
            # Validation
            if self.global_step % val_freq == 0:
                val_losses = self.validate(val_loader)
                
                for k, v in val_losses.items():
                    self.writer.add_scalar(f'val/{k}', v, self.global_step)
                
                print(f"\nStep {self.global_step} - Val losses: {val_losses}")
                
                # Save checkpoint
                self.save_checkpoint('latest')
                
                if val_losses['total'] < self.best_val_loss:
                    self.best_val_loss = val_losses['total']
                    self.save_checkpoint('best')
                    print(f"New best model saved with loss {self.best_val_loss:.4f}")
                
                # Scheduler step
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_losses['total'])
                    else:
                        self.scheduler.step()
        
        pbar.close()
        self.writer.close()
        print("Training completed!")


def main():
    parser = argparse.ArgumentParser(description='Train PROTAC Flow Matching model')
    parser.add_argument('--config', type=str, default='configs/train.yaml',
                        help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Log directory')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set seed
    seed_all(config['train']['seed'])
    
    # Create model
    model = ProTACFlowModel(
        num_atom_types=len(config['data']['atom_vocab']),
        num_bond_types=len(config['data']['bond_vocab']),
        h_dim=config['model']['h_dim'],
        edge_dim=config['model']['edge_dim'],
        cond_dim=config['model']['cond_dim'],
        time_emb_dim=config['model']['time_emb_dim'],
        n_layers=config['model']['n_layers'],
        num_heads=config['model']['num_heads'],
        N_L_max=config['data']['N_L_max'],
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Create data module
    data_module = ProtacDataModule(
        raw_path=config['data']['dataset_path'],
        processed_path=config['data']['processed_path'],
        N_L_max=config['data']['N_L_max'],
        batch_size=config['train']['batch_size'],
        num_workers=config['train']['num_workers'],
    )
    
    # Create trainer
    trainer = Trainer(
        config=config,
        model=model,
        data_module=data_module,
        device=args.device,
        log_dir=args.log_dir,
    )
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"Resumed from {args.resume}")
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()
