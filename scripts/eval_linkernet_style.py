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

import numpy as np
import torch
from rdkit import RDLogger
from tqdm.auto import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protac_fm.utils.reconstruct import eval_success_rate


def _load_results(sample_path: str, num_eval: int | None):
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
    args = parser.parse_args()

    RDLogger.DisableLog("rdApp.*")

    results, files = _load_results(args.sample_path, args.num_eval)
    if len(results) == 0:
        raise FileNotFoundError(f"No sampling results found under {args.sample_path}")

    per_item = []
    for f, r in zip(files, results):
        gen_mols = r.get("gen_mols", [])
        rr, cr = eval_success_rate(gen_mols)
        per_item.append({"file": os.path.basename(f), "recon_rate": rr, "complete_rate": cr, "num": len(gen_mols)})

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
        writer = csv.DictWriter(fp, fieldnames=["file", "num", "recon_rate", "complete_rate"])
        writer.writeheader()
        for row in per_item:
            writer.writerow(row)

        writer.writerow(
            {
                "file": "__MEAN__",
                "num": "",
                "recon_rate": float(np.mean(recon_rates)),
                "complete_rate": float(np.mean(complete_rates)),
            }
        )
        writer.writerow(
            {
                "file": "__MEDIAN__",
                "num": "",
                "recon_rate": float(np.median(recon_rates)),
                "complete_rate": float(np.median(complete_rates)),
            }
        )

    print(f"Saved summary to {out_csv}")


if __name__ == "__main__":
    main()
