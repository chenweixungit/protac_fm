"""Training script for PROTAC Flow Matching model.

Based on LinkerNet training script with same logging and checkpoint structure.
Usage: python scripts/train_fm_protac.py configs/training/fm_protac.yml --device cuda
"""
import argparse
import os
import sys
import time
import shutil

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from torch_geometric.data import Batch
from tqdm.auto import tqdm
from rdkit import RDLogger

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protac_fm.utils.misc import (
    load_config, seed_all, setup_logdir, get_optimizer, get_scheduler,
    inf_iterator, sum_weighted_losses, ValidationLossTape, log_losses
)
from protac_fm.data.linker_dataset import get_linker_dataset
from protac_fm.data.transforms import (
    FeaturizeAtom, BuildCompleteGraph, StackFragLocalPos, RelativeGeometry
)
from protac_fm.model.protac_flow import ProTACFlowModel
from protac_fm.utils.reconstruct import parse_sampling_result_with_bond, eval_success_rate

torch.multiprocessing.set_sharing_strategy('file_system')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default='./configs/training/fm_protac.yml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--train_report_iter', type=int, default=None)
    parser.add_argument('--sampling_iter', type=int, default=None)
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    RDLogger.DisableLog('rdApp.*')
    
    # Load configs
    config = load_config(args.config)
    seed_all(config.train.seed)
    
    # Setup log directory (auto-creates logs folder like LinkerNet)
    logger, writer, log_dir, ckpt_dir, vis_dir = setup_logdir(args.config, args.logdir)
    logger.info(args)
    logger.info(f'Config: {args.config}')
    logger.info(f'Log directory: {log_dir}')
    logger.info(f'Checkpoint directory: {ckpt_dir}')
    logger.info(f'Visualization directory: {vis_dir}')
    
    # Copy config to log directory
    shutil.copy(args.config, os.path.join(log_dir, 'config.yml'))
    
    # Get training report and sampling intervals from config or args
    train_report_iter = args.train_report_iter or config.train.get('train_report_iter', 200)
    sampling_iter = args.sampling_iter or config.train.get('sampling_iter', 10000)
    
    # Transforms
    cfg_model = config.model
    atom_featurizer = FeaturizeAtom(
        config.dataset.name,
        known_anchor=cfg_model.known_anchor
    )
    graph_builder = BuildCompleteGraph(
        known_linker_bond=cfg_model.known_linker_bond
    )
    max_num_atoms = config.dataset.get('max_num_atoms', 70)
    
    train_transform = Compose([
        atom_featurizer,
        graph_builder,
        StackFragLocalPos(max_num_atoms=max_num_atoms),
        RelativeGeometry(mode=cfg_model.rel_geometry)
    ])
    test_transform = Compose([
        atom_featurizer,
        graph_builder,
        StackFragLocalPos(max_num_atoms=max_num_atoms),
        RelativeGeometry(mode=cfg_model.rel_geometry)
    ])
    
    logger.info(f'Relative Geometry: {cfg_model.rel_geometry}')
    logger.info(f'Train transform: {train_transform}')
    
    # Datasets and loaders
    logger.info('Loading dataset...')
    dataset, subsets = get_linker_dataset(
        cfg=config.dataset,
        transform_map={'train': train_transform, 'val': test_transform, 'test': test_transform}
    )
    
    train_set, val_set, test_set = subsets['train'], subsets['val'], subsets['test']
    logger.info(f'Training: {len(train_set)} Validation: {len(val_set)} Test: {len(test_set)}')
    
    FOLLOW_BATCH = ['edge_type']
    COLLATE_EXCLUDE_KEYS = ['nbh_list']
    
    train_iterator = inf_iterator(DataLoader(
        train_set,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=COLLATE_EXCLUDE_KEYS
    ))
    val_loader = DataLoader(
        val_set, config.train.batch_size, shuffle=False,
        follow_batch=FOLLOW_BATCH, exclude_keys=COLLATE_EXCLUDE_KEYS
    )
    test_loader = DataLoader(
        test_set, config.train.batch_size, shuffle=False,
        follow_batch=FOLLOW_BATCH, exclude_keys=COLLATE_EXCLUDE_KEYS
    )
    
    # Model
    logger.info('Building model...')
    model = ProTACFlowModel(
        num_atom_types=atom_featurizer.num_classes,
        num_bond_types=graph_builder.num_bond_classes,
        h_dim=cfg_model.h_dim,
        edge_dim=cfg_model.edge_dim,
        cond_dim=cfg_model.cond_dim,
        time_emb_dim=cfg_model.time_emb_dim,
        n_layers=cfg_model.n_layers,
        num_heads=cfg_model.num_heads,
        N_L_max=cfg_model.N_L_max,
    ).to(args.device)
    
    logger.info('Num of parameters is %.2f M' % (np.sum([p.numel() for p in model.parameters()]) / 1e6))
    
    # Load checkpoint if specified
    if config.train.get('ckpt_path', None):
        ckpt = torch.load(config.train.ckpt_path, map_location=args.device)
        model.load_state_dict(ckpt['model'])
        logger.info(f'Loaded checkpoint from {config.train.ckpt_path}')
    
    # Optimizer and scheduler
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    
    def train(it, batch):
        model.train()
        optimizer.zero_grad()
        
        batch = batch.to(args.device)
        loss_dict = model.get_loss(batch, pos_noise_std=config.train.get('pos_noise_std', 0.))
        loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
        loss_dict['overall'] = loss
        
        loss.backward()
        orig_grad_norm = clip_grad_norm_(
            model.parameters(), config.train.max_grad_norm, error_if_nonfinite=True
        )
        optimizer.step()
        
        log_losses(loss_dict, it, 'train', train_report_iter, logger, writer, others={
            'grad': orig_grad_norm,
            'lr': optimizer.param_groups[0]['lr']
        })
    
    def validate(it):
        loss_tape = ValidationLossTape()
        with torch.no_grad():
            model.eval()
            for batch in tqdm(val_loader, desc='Validate', dynamic_ncols=True):
                for t in np.linspace(0, cfg_model.num_steps - 1, 10).astype(int):
                    time_step = torch.tensor([t] * batch.num_graphs).to(args.device)
                    batch = batch.to(args.device)
                    loss_dict = model.get_loss(batch, t=time_step)
                    loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
                    loss_dict['overall'] = loss
                    loss_tape.update(loss_dict, 1)
        
        avg_loss = loss_tape.log(it, logger, writer, 'val')
        
        if config.train.scheduler.type == 'plateau':
            scheduler.step(avg_loss)
        else:
            scheduler.step()
        
        return avg_loss
    
    def test(it):
        loss_tape = ValidationLossTape()
        with torch.no_grad():
            model.eval()
            for batch in tqdm(test_loader, desc='Test', dynamic_ncols=True):
                for t in np.linspace(0, cfg_model.num_steps - 1, 10).astype(int):
                    time_step = torch.tensor([t] * batch.num_graphs).to(args.device)
                    batch = batch.to(args.device)
                    loss_dict = model.get_loss(batch, t=time_step)
                    loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
                    loss_dict['overall'] = loss
                    loss_tape.update(loss_dict, 1)
        
        avg_loss = loss_tape.log(it, logger, writer, 'test')
        return avg_loss
    
    # Resume from checkpoint if specified
    start_iter = 1
    best_loss, best_iter = None, None
    
    if args.resume:
        logger.info(f'Resuming from checkpoint: {args.resume}')
        ckpt = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if ckpt.get('scheduler') and scheduler:
            scheduler.load_state_dict(ckpt['scheduler'])
        start_iter = ckpt.get('iteration', 0) + 1
        best_loss = ckpt.get('best_loss', None)
        best_iter = ckpt.get('best_iter', None)
        logger.info(f'Resumed from iteration {start_iter - 1}')
    
    # Training loop
    logger.info('='*60)
    logger.info('Starting training...')
    logger.info(f'Max iterations: {config.train.max_iters}')
    logger.info(f'Batch size: {config.train.batch_size}')
    logger.info(f'Validation frequency: {config.train.val_freq}')
    logger.info(f'Sampling frequency: {sampling_iter}')
    logger.info('='*60)
    
    try:
        model.train()
        
        for it in range(start_iter, config.train.max_iters + 1):
            t_start = time.time()
            batch = next(train_iterator)
            t_data = time.time() - t_start
            
            train(it, batch)
            
            # Sampling
            if it % sampling_iter == 0:
                try:
                    data = test_set[0]
                    data_list = [data.clone() for _ in range(100)]
                    batch = Batch.from_data_list(
                        data_list, follow_batch=FOLLOW_BATCH, exclude_keys=COLLATE_EXCLUDE_KEYS
                    ).to(args.device)
                    
                    with torch.no_grad():
                        model.eval()
                        results = model.sample(batch)
                        final_x = results['x_final']
                        final_c = results['atom_type_final']
                        final_bond = results['bond_type_final']
                    
                    gen_mols = parse_sampling_result_with_bond(
                        data_list, final_x, final_c, final_bond, atom_featurizer,
                        known_linker_bonds=cfg_model.known_linker_bond, check_validity=True
                    )
                    
                    save_path = os.path.join(vis_dir, f'sampling_results_{it}.pt')
                    torch.save({
                        'data': data,
                        'final_x': final_x, 'final_c': final_c, 'final_bond': final_bond,
                        'gen_mols': gen_mols
                    }, save_path)
                    logger.info(f'Saved sampling results to {save_path}')
                    
                    recon_rate, complete_rate = eval_success_rate(gen_mols)
                    logger.info(f'Recon rate: {recon_rate:.4f}')
                    logger.info(f'Complete rate: {complete_rate:.4f}')
                    writer.add_scalar('sampling/recon_rate', recon_rate, it)
                    writer.add_scalar('sampling/complete_rate', complete_rate, it)
                    writer.flush()
                    
                    model.train()
                except Exception as e:
                    logger.warning(f'Sampling failed at iter {it}: {e}')
                    model.train()
            
            # Validation
            if it % config.train.val_freq == 0 or it == config.train.max_iters:
                val_loss = validate(it)
                
                # Save checkpoint
                ckpt_path = os.path.join(ckpt_dir, f'{it}.pt')
                ckpt_data = {
                    'config': config,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict() if scheduler else None,
                    'iteration': it,
                    'best_loss': best_loss,
                    'best_iter': best_iter,
                }
                torch.save(ckpt_data, ckpt_path)
                logger.info(f'Saved checkpoint to {ckpt_path}')
                
                if best_loss is None or val_loss < best_loss:
                    logger.info(f'[Validate] Best val loss achieved: {val_loss:.6f}')
                    best_loss, best_iter = val_loss, it
                    
                    # Save best model
                    best_path = os.path.join(ckpt_dir, 'best.pt')
                    ckpt_data['best_loss'] = best_loss
                    ckpt_data['best_iter'] = best_iter
                    torch.save(ckpt_data, best_path)
                    logger.info(f'Saved best model to {best_path}')
                    
                    # Run test on best model
                    test(it)
                else:
                    logger.info(f'[Validate] Current: {val_loss:.6f}, Best: {best_loss:.6f} at iter {best_iter}')
                
                model.train()
    
    except KeyboardInterrupt:
        logger.info('Training interrupted by user')
        # Save interrupt checkpoint
        interrupt_path = os.path.join(ckpt_dir, 'interrupt.pt')
        torch.save({
            'config': config,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None,
            'iteration': it,
            'best_loss': best_loss,
            'best_iter': best_iter,
        }, interrupt_path)
        logger.info(f'Saved interrupt checkpoint to {interrupt_path}')
    
    except Exception as e:
        logger.error(f'Training failed with error: {e}')
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    # Final summary
    logger.info('='*60)
    logger.info('TRAINING COMPLETED')
    logger.info('='*60)
    if best_loss is not None:
        logger.info(f'Best iteration: {best_iter}')
        logger.info(f'Best validation loss: {best_loss:.6f}')
    logger.info(f'Checkpoints saved to: {ckpt_dir}')
    logger.info(f'Logs saved to: {log_dir}')
    logger.info('='*60)
    writer.close()


if __name__ == '__main__':
    main()
