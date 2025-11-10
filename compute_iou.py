import os
import pickle
import argparse
import math
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute IoU overlap of top-X% attribution nodes across fixed methods"
    )
    p.add_argument(
        "--dataset", "-d", type=str, required=True,
        help="Name of the subfolder under 'attributions/' (e.g. 'KIBA' or 'GLASS')"
    )
    p.add_argument(
        "--top-percent", "-p", type=float, default=50.0,
        help="Top percentage of nodes to select by attribution (0–100]. Default: 50"
    )
    return p.parse_args()


script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
ATTS_ROOT    = os.path.join(project_root, "attributions")


def load_attributions(dataset, method_name):

    file_path = os.path.join(ATTS_ROOT, dataset, f"{method_name}.pkl")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"[DEBUG] Loading '{method_name}.pkl' from '{dataset}/' …")
    with open(file_path, "rb") as f:
        records = pickle.load(f)

    arrays = [np.asarray(rec["attributions"]) for rec in records]
    print(f"[DEBUG]   Loaded {len(arrays)} graphs for '{method_name}'")
    return arrays


def select_top_fraction_nodes(scores: np.ndarray, top_frac: float):

    N = scores.shape[0]
    if N == 0:
        return set()
    if top_frac <= 0.0:
        return set()
    k = max(1, math.ceil(top_frac * N))
    top_idxs = np.argpartition(scores, -k)[-k:]
    return set(top_idxs)


def compute_iou_across_methods(selected_dict):

    methods = list(selected_dict.keys())
    if not methods:
        return []

    lengths = [len(selected_dict[m]) for m in methods]
    if len(set(lengths)) != 1:
        raise ValueError(f"Mismatch in number of graphs across methods: {dict(zip(methods, lengths))}")

    n_graphs = lengths[0]
    ious = []
    for i in range(n_graphs):
        sets = [selected_dict[m][i] for m in methods]
        inter = set.intersection(*sets) if sets else set()
        union = set.union(*sets) if sets else set()
        iou = (len(inter) / len(union)) if union else 0.0
        ious.append(iou)
    return ious


if __name__ == "__main__":
    args    = parse_args()
    DS      = args.dataset
    METHODS = ["integrated", "inputxgrad", "guided", "shap"]

    pct = args.top_percent
    if pct < 0 or pct > 100:
        print(f"[WARN] --top-percent {pct} out of range; clamping to [0,100].")
    pct = min(100.0, max(0.0, pct))
    top_frac = pct / 100.0

    print(f"[INFO] Dataset={DS} | Top={pct:.2f}% of nodes per method")

    all_atts = {m: load_attributions(DS, m) for m in METHODS}


    selected = {}
    for m, arrs in all_atts.items():
        print(f"[DEBUG] Selecting top {pct:.2f}% for '{m}' …")
        selected[m] = [select_top_fraction_nodes(arr, top_frac) for arr in arrs]

    iou_list = compute_iou_across_methods(selected)
    iou_arr = np.asarray(iou_list, dtype=float)
    n = iou_arr.size

    mean_iou = np.mean(iou_arr) if n > 0 else float("nan")
    sd_iou   = (np.std(iou_arr, ddof=1) if n > 1 else float("nan"))   # sample SD
    sem_iou  = (sd_iou / np.sqrt(n)     if n > 1 else float("nan"))

    print(f"\nDataset: {DS}")
    print(f"Graphs: {n}")
    print(f"Top fraction: {top_frac:.3f} ({pct:.2f}%)")
    print(f"IoU mean ± SD: {mean_iou:.4f} ± {sd_iou:.4f}")
    print(f"IoU SEM: {sem_iou:.4f}" if n > 1 else "IoU SEM: N/A (n < 2)")
