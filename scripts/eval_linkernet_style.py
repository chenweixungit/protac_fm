"""Evaluation script (LinkerNet-style) for PROTAC sampling outputs.

Behavior mirrors LinkerNet/scripts/eval_protac.py:
- Accept a single .pt file OR a directory containing sampling_*.pt
- Compute quick reconstruction/completeness rates
- Save a summary CSV (default: <sample_path>/summary.csv)

Usage:
  python scripts/eval_linkernet_style.py <sample_path_or_pt> --save True
"""

import argparse
import csv
import os
import sys
from glob import glob
from typing import Optional

import numpy as np
import torch
from rdkit import RDLogger
from rdkit import Chem
from tqdm.auto import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protac_fm.utils.reconstruct import eval_success_rate


def _count_recon_complete(gen_mols):
    recon = 0
    complete = 0
    total = len(gen_mols)
    for mol in gen_mols:
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            continue
        recon += 1
        smiles = Chem.MolToSmiles(mol)
        if "." not in smiles:
            complete += 1
    return recon, complete, total


def _as_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x)


def _analyze_hit_rates(result_dict):
    """Analyze whether predicted bonds connect linker to frag1 and/or frag2.

    Returns:
        (hit_f1_rate, hit_f2_rate, hit_both_rate)
    """
    if "data_list" not in result_dict or "final_bond" not in result_dict:
        return None

    data_list = result_dict["data_list"]
    final_bond = result_dict["final_bond"]

    if isinstance(final_bond, list):
        fb = torch.stack([_as_tensor(x).long() for x in final_bond], dim=0)
    else:
        fb = _as_tensor(final_bond).long()
        if fb.dim() == 1:
            fb = fb.unsqueeze(0)

    S = min(len(data_list), fb.size(0))
    if S == 0:
        return None

    hit_f1 = 0
    hit_f2 = 0
    hit_both = 0

    for i in range(S):
        d = data_list[i]
        edge_index = _as_tensor(d.edge_index).long()
        fragment_mask = _as_tensor(d.fragment_mask).long()
        linker_mask = (fragment_mask == 0)

        src = edge_index[0]
        dst = edge_index[1]

        cross = (linker_mask[src] & (fragment_mask[dst] > 0)) | (linker_mask[dst] & (fragment_mask[src] > 0))
        nz = (fb[i] > 0) & cross
        if not nz.any():
            continue

        edges = nz.nonzero(as_tuple=True)[0]
        u = src[edges]
        v = dst[edges]

        # whether any predicted cross edge touches frag1/frag2
        touch_f1 = ((fragment_mask[u] == 1) | (fragment_mask[v] == 1)).any().item()
        touch_f2 = ((fragment_mask[u] == 2) | (fragment_mask[v] == 2)).any().item()

        if touch_f1:
            hit_f1 += 1
        if touch_f2:
            hit_f2 += 1
        if touch_f1 and touch_f2:
            hit_both += 1

    return hit_f1 / S, hit_f2 / S, hit_both / S


def _load_results(sample_path: str, num_eval: Optional[int]):
    if sample_path.endswith(".pt"):
        return [torch.load(sample_path)], [sample_path]

    results_file_list = sorted(glob(os.path.join(sample_path, "sampling_*.pt")))
    if num_eval is not None:
        results_file_list = results_file_list[:num_eval]

    results = []
    for f in tqdm(results_file_list, desc="Load sampling files"):
        results.append(torch.load(f))

    return results, results_file_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_path", type=str)
    parser.add_argument("--num_eval", type=int, default=None)
    parser.add_argument("--save", type=eval, default=True)
    parser.add_argument("--out_csv", type=str, default=None)
    parser.add_argument("--analyze_connectivity", type=eval, default=False)
    args = parser.parse_args()

    RDLogger.DisableLog("rdApp.*")

    results, files = _load_results(args.sample_path, args.num_eval)
    if len(results) == 0:
        raise FileNotFoundError(f"No sampling results found under {args.sample_path}")

    per_item = []
    for f, r in zip(files, results):
        gen_mols = r.get("gen_mols", [])
        recon_cnt, complete_cnt, total_cnt = _count_recon_complete(gen_mols)
        rr = recon_cnt / total_cnt if total_cnt > 0 else 0.0
        cr = complete_cnt / total_cnt if total_cnt > 0 else 0.0

        row = {
            "file": os.path.basename(f),
            "num": total_cnt,
            "recon_rate": rr,
            "complete_rate": cr,
            "recon_count": recon_cnt,
            "complete_count": complete_cnt,
        }
        if args.analyze_connectivity:
            hit = _analyze_hit_rates(r)
            if hit is not None:
                row["hit_f1_rate"], row["hit_f2_rate"], row["hit_both_rate"] = hit
            else:
                row["hit_f1_rate"], row["hit_f2_rate"], row["hit_both_rate"] = "", "", ""

        per_item.append(row)

    recon_rates = [x["recon_rate"] for x in per_item]
    complete_rates = [x["complete_rate"] for x in per_item]

    print(f"recon rate:    mean: {np.mean(recon_rates):.4f} median: {np.median(recon_rates):.4f}")
    print(f"complete rate: mean: {np.mean(complete_rates):.4f} median: {np.median(complete_rates):.4f}")

    if not args.save:
        return

    if args.out_csv is not None:
        out_csv = args.out_csv
    else:
        out_csv = os.path.join(args.sample_path, "summary.csv") if not args.sample_path.endswith(".pt") else os.path.join(os.path.dirname(args.sample_path), "summary.csv")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    with open(out_csv, "w", newline="", encoding="utf-8") as fp:
        fieldnames = ["file", "num", "recon_rate", "complete_rate", "recon_count", "complete_count"]
        if args.analyze_connectivity:
            fieldnames += ["hit_f1_rate", "hit_f2_rate", "hit_both_rate"]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_item:
            writer.writerow(row)

        mean_row = {
            "file": "__MEAN__",
            "num": "",
            "recon_rate": float(np.mean(recon_rates)),
            "complete_rate": float(np.mean(complete_rates)),
            "recon_count": "",
            "complete_count": "",
        }
        median_row = {
            "file": "__MEDIAN__",
            "num": "",
            "recon_rate": float(np.median(recon_rates)),
            "complete_rate": float(np.median(complete_rates)),
            "recon_count": "",
            "complete_count": "",
        }
        if args.analyze_connectivity:
            mean_row.update({"hit_f1_rate": "", "hit_f2_rate": "", "hit_both_rate": ""})
            median_row.update({"hit_f1_rate": "", "hit_f2_rate": "", "hit_both_rate": ""})

        writer.writerow(mean_row)
        writer.writerow(median_row)

    print(f"Saved summary to {out_csv}")


if __name__ == "__main__":
    main()
