"""Sampling script (LinkerNet-style) for PROTAC Flow Matching model.

Outputs per-test-item files:
  sampling_000000.pt, sampling_000001.pt, ...
Each file contains keys consistent with LinkerNet:
  - ref_data
  - data_list
  - final_x / final_c / final_bond
  - gen_mols

Usage:
  python scripts/sample_linkernet_style.py configs/training/fm_protac.yml --ckpt_path <ckpt> --subset test
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from rdkit import RDLogger
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose
from tqdm.auto import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protac_fm.data.linker_dataset import get_linker_dataset
from protac_fm.data.transforms import BuildCompleteGraph, FeaturizeAtom, RelativeGeometry, StackFragLocalPos
from protac_fm.model.protac_flow import ProTACFlowModel
from protac_fm.utils.misc import load_config, seed_all, setup_logdir
from protac_fm.utils.reconstruct import eval_success_rate, parse_sampling_result_with_bond


torch.multiprocessing.set_sharing_strategy("file_system")


def _strip_rdkit_objs(data):
    for k in ["rdmol", "mol", "frag_mol", "link_mol"]:
        try:
            if hasattr(data, k):
                delattr(data, k)
        except Exception:
            pass
        try:
            if k in data:
                data.pop(k)
        except Exception:
            pass
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--subset", choices=["val", "test"], default="val")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--end_id", type=int, default=-1)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--outdir", type=str, default="./outputs_test")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--save_traj", type=eval, default=False)
    parser.add_argument("--n_steps", type=int, default=None)
    parser.add_argument("--skip_reconstruct", type=eval, default=False)
    parser.add_argument("--check_validity", type=eval, default=True)
    parser.add_argument("--prune_cross_bonds", type=eval, default=False)
    parser.add_argument("--max_cross_per_frag", type=int, default=1)
    parser.add_argument("--max_total_cross", type=int, default=2)
    parser.add_argument("--save_rdkit", type=eval, default=True)
    args = parser.parse_args()

    RDLogger.DisableLog("rdApp.*")

    config = load_config(args.config)
    if "sample" in config and hasattr(config.sample, "seed") and config.sample.seed is not None:
        seed = config.sample.seed
    elif "train" in config and hasattr(config.train, "seed") and config.train.seed is not None:
        seed = config.train.seed
    else:
        seed = 2023
    seed_all(seed)

    logger, writer, log_dir, ckpt_dir, vis_dir = setup_logdir(
        args.config, args.outdir
    )
    logger.info(args)

    ckpt = torch.load(args.ckpt_path, map_location=args.device)
    cfg_model = ckpt.get("configs", config).model

    atom_featurizer = FeaturizeAtom(config.dataset.name, known_anchor=cfg_model.known_anchor)
    graph_builder = BuildCompleteGraph(known_linker_bond=cfg_model.known_linker_bond)
    max_num_atoms = config.dataset.get("max_num_atoms", 70)

    init_transform = Compose(
        [
            atom_featurizer,
            graph_builder,
            StackFragLocalPos(max_num_atoms=max_num_atoms),
            RelativeGeometry(mode=getattr(cfg_model, "rel_geometry", "two_pos_and_rot")),
        ]
    )

    logger.info("Loading dataset...")
    dataset, subsets = get_linker_dataset(
        cfg=config.dataset,
        transform_map={"train": None, "val": None, "test": None},
    )
    test_set = subsets[args.subset]
    logger.info(f"Subset {args.subset}: {len(test_set)}")

    logger.info("Building model...")
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
        num_steps=cfg_model.num_steps,
    ).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    FOLLOW_BATCH = ["edge_type"]
    COLLATE_EXCLUDE_KEYS = ["nbh_list", "rdmol", "mol", "frag_mol", "link_mol"]

    if args.end_id == -1:
        args.end_id = len(test_set)

    n_steps = args.n_steps or config.sample.get("n_steps", 100)

    logger.info(
        f"Start sampling subset={args.subset} range=[{args.start_id},{args.end_id}) "
        f"num_samples={args.num_samples} n_steps={n_steps} device={args.device}"
    )

    for idx in tqdm(range(args.start_id, args.end_id)):
        t0 = time.time()
        raw_data = test_set[idx]

        t1 = time.time()
        data_list = [raw_data.clone() for _ in range(args.num_samples)]
        new_data_list = [init_transform(d) for d in data_list]

        t2 = time.time()

        batch = Batch.from_data_list(
            new_data_list,
            follow_batch=FOLLOW_BATCH,
            exclude_keys=COLLATE_EXCLUDE_KEYS,
        ).to(args.device)

        t3 = time.time()

        with torch.no_grad():
            results = model.sample(batch, n_steps=n_steps)
            final_x = results["x_final"]
            final_c = results["atom_type_final"]
            final_bond = results["bond_type_final"]

        t4 = time.time()

        if args.skip_reconstruct:
            gen_mols = []
        else:
            gen_mols = parse_sampling_result_with_bond(
                new_data_list,
                final_x,
                final_c,
                final_bond,
                atom_featurizer,
                known_linker_bonds=cfg_model.known_linker_bond,
                check_validity=args.check_validity,
                prune_cross_bonds=args.prune_cross_bonds,
                max_cross_per_frag=args.max_cross_per_frag,
                max_total_cross=args.max_total_cross,
            )

        t5 = time.time()

        save_path = os.path.join(log_dir, f"sampling_{idx:06d}.pt")
        ref_data = init_transform(raw_data.clone())
        data_list_to_save = [d.clone() for d in new_data_list]
        if not args.save_rdkit:
            _strip_rdkit_objs(ref_data)
            for d in data_list_to_save:
                _strip_rdkit_objs(d)

        save_dict = {
            "ref_data": ref_data,
            "data_list": data_list_to_save,
            "final_x": final_x,
            "final_c": final_c,
            "final_bond": final_bond,
            "gen_mols": gen_mols,
        }
        if args.save_traj and "traj" in results:
            save_dict["traj"] = results["traj"]

        torch.save(save_dict, save_path)

        t6 = time.time()
        logger.info(
            f"idx={idx} time(s): load={t1 - t0:.3f} "
            f"transform={t2 - t1:.3f} batch={t3 - t2:.3f} "
            f"sample={t4 - t3:.3f} reconstruct={t5 - t4:.3f} save={t6 - t5:.3f}"
        )

    logger.info("Sample done!")

    recon_rate, complete_rate = [], []
    for idx in range(args.start_id, args.end_id):
        load_path = os.path.join(log_dir, f"sampling_{idx:06d}.pt")
        results = torch.load(load_path)
        rr, cr = eval_success_rate(results["gen_mols"])
        logger.info(f"idx: {idx} recon rate: {rr} complete rate: {cr}")
        recon_rate.append(rr)
        complete_rate.append(cr)

    logger.info(f"recon rate:    mean: {np.mean(recon_rate):.4f} median: {np.median(recon_rate):.4f}")
    logger.info(f"complete rate: mean: {np.mean(complete_rate):.4f} median: {np.median(complete_rate):.4f}")


if __name__ == "__main__":
    main()
