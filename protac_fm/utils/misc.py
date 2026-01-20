"""Miscellaneous utilities."""
import os
import logging
import random
import yaml
from datetime import datetime
from easydict import EasyDict

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


def load_config(config_path: str) -> EasyDict:
    """Load YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return EasyDict(config)


def seed_all(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logdir(config_path: str, logdir: str):
    """Setup logging directory and logger.
    
    Automatically creates logs directory if it doesn't exist (like LinkerNet).
    """
    # Ensure root logdir exists (auto-create like LinkerNet)
    os.makedirs(logdir, exist_ok=True)
    
    config_name = os.path.basename(config_path).replace('.yml', '').replace('.yaml', '')
    timestamp = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    log_dir = os.path.join(logdir, f'{config_name}_{timestamp}')
    
    # Create log directory and subdirectories
    os.makedirs(log_dir, exist_ok=True)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    vis_dir = os.path.join(log_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Setup logger
    logger = logging.getLogger('train')
    logger.setLevel(logging.DEBUG)
    
    # File handler
    fh = logging.FileHandler(os.path.join(log_dir, 'train.log'))
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    # TensorBoard writer
    writer = SummaryWriter(os.path.join(log_dir, 'tensorboard'))
    
    return logger, writer, log_dir, ckpt_dir, vis_dir


def get_optimizer(config, model):
    """Create optimizer from config."""
    if config.type == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.get('weight_decay', 0),
            betas=(config.get('beta1', 0.9), config.get('beta2', 0.999))
        )
    elif config.type == 'adamw':
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.get('weight_decay', 0),
            betas=(config.get('beta1', 0.9), config.get('beta2', 0.999))
        )
    else:
        raise ValueError(f'Unknown optimizer type: {config.type}')


def get_scheduler(config, optimizer):
    """Create scheduler from config."""
    if config.type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.factor,
            patience=config.patience,
            min_lr=config.min_lr
        )
    elif config.type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.T_max,
            eta_min=config.get('min_lr', 0)
        )
    elif config.type == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.step_size,
            gamma=config.gamma
        )
    else:
        return None


def inf_iterator(iterable):
    """Infinite iterator."""
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


def sum_weighted_losses(loss_dict, weights):
    """Sum losses with weights."""
    total = 0.0
    for key, loss in loss_dict.items():
        weight = weights.get(key, 1.0)
        total = total + weight * loss
    return total


class ValidationLossTape:
    """Tape for recording validation losses."""
    
    def __init__(self):
        self.losses = {}
        self.counts = {}
    
    def update(self, loss_dict, count=1):
        for key, loss in loss_dict.items():
            if key not in self.losses:
                self.losses[key] = 0.0
                self.counts[key] = 0
            self.losses[key] += loss.item() if torch.is_tensor(loss) else loss
            self.counts[key] += count
    
    def log(self, it, logger, writer, prefix='val'):
        avg_losses = {}
        for key in self.losses:
            avg_losses[key] = self.losses[key] / self.counts[key]
            writer.add_scalar(f'{prefix}/{key}', avg_losses[key], it)
        
        loss_str = ' '.join([f'{k}: {v:.6f}' for k, v in avg_losses.items()])
        logger.info(f'[{prefix.upper()}] Iter {it} {loss_str}')
        
        return avg_losses.get('overall', sum(avg_losses.values()) / len(avg_losses))


def log_losses(loss_dict, it, prefix, log_freq, logger, writer, others=None):
    """Log losses to logger and tensorboard."""
    if it % log_freq == 0:
        loss_str = ' '.join([f'{k}: {v.item() if torch.is_tensor(v) else v:.6f}' for k, v in loss_dict.items()])
        if others:
            loss_str += ' ' + ' '.join([f'{k}: {v:.6f}' for k, v in others.items()])
        logger.info(f'[{prefix.upper()}] Iter {it} {loss_str}')
        
        for key, loss in loss_dict.items():
            writer.add_scalar(f'{prefix}/{key}', loss.item() if torch.is_tensor(loss) else loss, it)
        if others:
            for key, val in others.items():
                writer.add_scalar(f'{prefix}/{key}', val, it)
