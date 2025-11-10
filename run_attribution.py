#!/usr/bin/env python3
import argparse
import os
import sys
import yaml
import pickle
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm


proj_root = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from rindti.data import DTIDataset
from rindti.models import ClassificationModel, RegressionModel

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute node attributions for DTI model (any subset of methods)"
    )
    p.add_argument("--config",     type=str, required=True,
                   help="YAML config (has datamodule & model sections)")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to .ckpt")
    p.add_argument("--methods",    type=str, nargs="+",
                   choices=["integrated","inputxgrad","guided","shap"],
                   default=["integrated","inputxgrad","guided","shap"],
                   help="One or more attribution methods to run")
    p.add_argument("--split",      type=str, default="test",
                   choices=["train","val","test"],
                   help="Which split to run on")
    p.add_argument("--batch_size", type=int, default=4,
                   help="DataLoader batch size (>=1)")
    p.add_argument("--num_workers", type=int, default=4,
                   help="Number of DataLoader workers")
    p.add_argument("--device",     type=str, default="cuda",
                   help="Torch device")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory to write per-method .pkl files")
    return p.parse_args()

def get_model_class(mod_str):
    if mod_str == "class":
        return ClassificationModel
    if mod_str == "reg":
        return RegressionModel
    raise ValueError(f"Unknown model.module: {mod_str}")


def integrated_gradients(model, dp, steps=50, device="cuda"):
    model.eval()
    dx = dp.drug_x.to(device)
    px = dp.prot_x.to(device)
    baseline = torch.zeros_like(dx, device=device)
    alphas = torch.linspace(0, 1, steps+1, device=device).view(-1, 1, 1)
    interps = (baseline.unsqueeze(0) + alphas * (dx.unsqueeze(0) - baseline.unsqueeze(0))).requires_grad_()
    xi = interps.view(-1, dx.size(-1))
    pi = px.unsqueeze(0).repeat(steps+1, 1, 1).view(-1, px.size(-1))
    b_nodes = dx.size(0)
    p_nodes = px.size(0)
    idxs = torch.arange(steps+1, device=device)
    d_batch = idxs.repeat_interleave(b_nodes)
    p_batch = idxs.repeat_interleave(p_nodes)
    de_list, pe_list = [], []
    for k in range(steps+1):
        de_list.append(dp.drug_edge_index.to(device) + k * b_nodes)
        pe_list.append(dp.prot_edge_index.to(device) + k * p_nodes)
    de = torch.cat(de_list, dim=1)
    pe = torch.cat(pe_list, dim=1)
    with autocast():
        out = model(
            {"x": pi,  "edge_index": pe,  "batch": p_batch},
            {"x": xi,  "edge_index": de,  "batch": d_batch},
        )["pred"][:, 0]
    grads = torch.autograd.grad(out, interps, grad_outputs=torch.ones_like(out))[0]
    grads = grads.view(steps+1, b_nodes, dx.size(-1))

    avg_grads = (grads[:-1] + grads[1:]).mul(0.5).sum(dim=0).div(steps)
    attributions = (dx - baseline) * avg_grads
    return attributions.sum(dim=1).detach().cpu().numpy()


def inputxgrad(model, dp, device="cuda"):
    model.eval()
    dx = dp.drug_x.to(device).requires_grad_()
    px = dp.prot_x.to(device)
    with autocast():
        out = model(
            {"x": px, "edge_index": dp.prot_edge_index.to(device),
             "batch": torch.zeros(px.size(0), dtype=torch.long, device=device)},
            {"x": dx, "edge_index": dp.drug_edge_index.to(device),
             "batch": torch.zeros(dx.size(0), dtype=torch.long, device=device)},
        )["pred"][0, 0]
    g = torch.autograd.grad(out, dx)[0]
    return (dx * g).sum(dim=1).detach().cpu().numpy()


def guided_backprop(model, dp, device="cuda"):
    class GB:
        def __init__(self, m):
            self.hooks = []
            for n, m_ in m.named_modules():
                if isinstance(m_, (torch.nn.ReLU, torch.nn.PReLU)) and "drug" in n.lower():
                    self.hooks.append(m_.register_backward_hook(self.hook))
        def hook(self, mod, grad_in, grad_out):
            if grad_in[0] is not None:
                return (torch.clamp(grad_in[0], min=0.0),) + grad_in[1:]
        def close(self):
            for h in self.hooks:
                h.remove()

    model.eval()
    gb = GB(model)
    dx = dp.drug_x.to(device).requires_grad_()
    px = dp.prot_x.to(device)
    with autocast():
        out = model(
            {"x": px, "edge_index": dp.prot_edge_index.to(device),
             "batch": torch.zeros(px.size(0), dtype=torch.long, device=device)},
            {"x": dx, "edge_index": dp.drug_edge_index.to(device),
             "batch": torch.zeros(dx.size(0), dtype=torch.long, device=device)},
        )["pred"][0, 0]
    model.zero_grad()
    out.backward()
    gb.close()
    return (dx * dx.grad).sum(dim=1).detach().cpu().numpy()


def gradient_shap(model, dp, steps=20, baseline_samples=5,
                  noise_std=0.1, device="cuda"):
    model.eval()
    dx = dp.drug_x.to(device)
    px = dp.prot_x.to(device)

    baselines = torch.zeros_like(dx).unsqueeze(0).repeat(baseline_samples, 1, 1).to(device)
    if noise_std > 0:
        baselines += noise_std * torch.randn_like(baselines)

    all_attr = []
    for b in baselines:
        alphas = torch.linspace(0, 1, steps+1, device=device).view(-1, 1, 1)
        interps = (b + alphas * (dx.unsqueeze(0) - b)).requires_grad_()
        xi = interps.view(-1, dx.size(-1))
        pi = px.unsqueeze(0).repeat(steps+1, 1, 1).view(-1, px.size(-1))
        bi = torch.arange(steps+1, device=device)
        d_batch = bi.repeat_interleave(dx.size(0))
        p_batch = bi.repeat_interleave(px.size(0))

        de_list, pe_list = [], []
        for k in range(steps+1):
            de_list.append(dp.drug_edge_index.to(device) + k * dx.size(0))
            pe_list.append(dp.prot_edge_index.to(device) + k * px.size(0))
        de = torch.cat(de_list, dim=1)
        pe = torch.cat(pe_list, dim=1)

        with autocast():
            out = model(
                {"x": pi, "edge_index": pe, "batch": p_batch},
                {"x": xi, "edge_index": de, "batch": d_batch}
            )["pred"][..., 0]
        grad = torch.autograd.grad(out, interps, grad_outputs=torch.ones_like(out))[0]
        grad = grad.view(steps+1, dx.size(0), dx.size(-1))
        avg_grad = (grad[:-1] + grad[1:]).mul(0.5).sum(dim=0).div(steps)
        all_attr.append(((dx - b) * avg_grad).sum(dim=1))

    return torch.stack(all_attr, dim=0).mean(dim=0).detach().cpu().numpy()


def main():
    args = parse_args()

    # load config & model
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["model"]
    data_cfg = cfg["datamodule"]
    ModelClass = get_model_class(model_cfg["module"])
    model = ModelClass.load_from_checkpoint(
        args.checkpoint, **model_cfg
    ).to(args.device)
    model.eval()

    # dataset + loader
    ds = DTIDataset(
        filename=data_cfg["filename"],
        exp_name=data_cfg["exp_name"],
        split=args.split
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    methods = {
        "integrated": integrated_gradients,
        "inputxgrad": inputxgrad,
        "guided": guided_backprop,
        "shap": gradient_shap,
    }

    os.makedirs(args.output_dir, exist_ok=True)

    for method in args.methods:
        fn = methods[method]
        records = []
        pbar = tqdm(total=len(ds), desc=method, unit="sample")
        idx = 0
        for batch in loader:
            dps = batch.to_data_list() if hasattr(batch, "to_data_list") else [batch]
            for dp in dps:
                scores = fn(model, dp, device=args.device)
                records.append({
                    "index": idx,
                    "drug_id": dp.drug_id,
                    "prot_id": dp.prot_id,
                    "attributions": scores,
                })
                idx += 1
                pbar.update(1)
        pbar.close()

        out_path = os.path.join(args.output_dir, f"{method}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(records, f)

if __name__ == "__main__":
    main()
