#!/usr/bin/env python3
"""
train_lottomax_set.py

Option B (closest to Lotto Max mechanics):
- Treat the 7 main numbers as an UNORDERED SET (multi-label prediction over 50 numbers)
- Treat the bonus number separately (single-label over 50), with a constraint:
  bonus cannot be one of the 7 main numbers (we mask those logits during loss/metrics)

Distributed training:
- Works on 1..N GPUs via torchrun (DDP).
- For 2 GPUs: torchrun --standalone --nproc_per_node=2 train_lottomax_set.py --data_path lottomax.csv

Data format (CSV header required):
date,n1,n2,n3,n4,n5,n6,n7,bonus
2025-12-23,3,28,37,38,39,41,43,45

Notes:
- This is a "next draw" predictor over a sequence of past draws.
- If the underlying process is truly random, out-of-sample performance should match chance.
- This harness is still useful for measuring bias/leakage/non-random structure.
"""

import argparse
import csv
import math
import os
import random
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler


# ---------------------------
# DDP utilities
# ---------------------------

def ddp_is_enabled() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()

def ddp_rank() -> int:
    return torch.distributed.get_rank() if ddp_is_enabled() else 0

def ddp_world_size() -> int:
    return torch.distributed.get_world_size() if ddp_is_enabled() else 1

def ddp_barrier():
    if ddp_is_enabled():
        torch.distributed.barrier()

@torch.no_grad()
def all_reduce_mean(x: torch.Tensor) -> torch.Tensor:
    if not ddp_is_enabled():
        return x
    y = x.clone()
    torch.distributed.all_reduce(y, op=torch.distributed.ReduceOp.SUM)
    y /= ddp_world_size()
    return y

def log_rank0(msg: str):
    if ddp_rank() == 0:
        print(msg, flush=True)

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------
# Data loading
# ---------------------------

def load_lottomax_csv(path: str):
    """
    Loads Lotto Max CSV/TSV with auto-detected delimiter.

    Required columns:
      date, n1, n2, n3, n4, n5, n6, n7, bonus

    Numbers must be 1..50 in the file; they are converted to 0..49 internally.

    Returns:
      List of tuples: [([n1..n7], bonus), ...]
    """
    import csv

    draws = []

    with open(path, "r", encoding="utf-8") as f:
        # --- auto-detect delimiter (comma, tab, or semicolon) ---
        sample = f.read(4096)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        reader = csv.DictReader(f, dialect=dialect)

        required_cols = ["n1", "n2", "n3", "n4", "n5", "n6", "n7", "bonus"]
        for col in required_cols:
            if col not in reader.fieldnames:
                raise ValueError(
                    f"CSV missing required column '{col}'. "
                    f"Found columns: {reader.fieldnames}"
                )

        for row in reader:
            main = [int(row[f"n{i}"]) for i in range(1, 8)]
            bonus = int(row["bonus"])

            # Convert 1..50 → 0..49
            main0 = [x - 1 for x in main]
            bonus0 = bonus - 1

            # Validation
            if len(set(main0)) != 7:
                raise ValueError(f"Duplicate main numbers found: {main}")
            if bonus0 in main0:
                raise ValueError(f"Bonus overlaps main numbers: {main} + {bonus}")

            for x in main0 + [bonus0]:
                if not (0 <= x < 50):
                    raise ValueError(f"Number out of range after conversion: {x}")

            draws.append((main0, bonus0))

    if len(draws) < 200:
        print(f"Warning: only {len(draws)} draws loaded — model may underfit.")

    return draws



def draw_to_feature(main0: List[int], bonus0: int) -> torch.Tensor:
    """
    Feature vector for a single draw:
      - 50-dim multi-hot for main set
      - 50-dim one-hot for bonus
    Total: 100 dims.
    """
    x = torch.zeros(100, dtype=torch.float32)
    for m in main0:
        x[m] = 1.0
    x[50 + bonus0] = 1.0
    return x


class NextDrawDataset(Dataset):
    """
    Builds (context -> next draw) examples.

    Input:
      - context_len draws as features [T, 100]
    Target:
      - main multi-hot [50]
      - bonus index scalar (0..49)
    """
    def __init__(self, draws: List[Tuple[List[int], int]], context_len: int):
        if len(draws) <= context_len:
            raise ValueError("Not enough draws for the requested context_len.")
        self.draws = draws
        self.context_len = context_len

        # Precompute features for speed
        self.features = torch.stack([draw_to_feature(m, b) for (m, b) in draws], dim=0)  # [N,100]

    def __len__(self) -> int:
        return len(self.draws) - self.context_len

    def __getitem__(self, idx: int):
        # context draws: idx .. idx+T-1, target is idx+T
        T = self.context_len
        context = self.features[idx: idx + T]  # [T,100]
        main, bonus = self.draws[idx + T]

        target_main = torch.zeros(50, dtype=torch.float32)
        target_main[main] = 1.0
        target_bonus = torch.tensor(bonus, dtype=torch.long)
        return context, target_main, target_bonus


# ---------------------------
# Model: Transformer over draws (not over numbers)
# ---------------------------

class DrawTransformer(nn.Module):
    """
    Transformer encodes a sequence of past draws (each draw is a 100-dim vector).
    Outputs:
      - logits_main: [B, 50] multi-label logits (we choose top-7 at evaluation)
      - logits_bonus: [B, 50] single-label logits (masked to exclude predicted/true main numbers)
    """
    def __init__(self, context_len: int, d_model: int = 512, nhead: int = 8, num_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        self.context_len = context_len
        self.in_proj = nn.Linear(100, d_model)
        self.pos_emb = nn.Embedding(context_len, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model)

        self.head_main = nn.Linear(d_model, 50)   # multi-label
        self.head_bonus = nn.Linear(d_model, 50)  # single-label

        # Causal mask so the model can't "peek" ahead inside the context window
        mask = torch.triu(torch.ones(context_len, context_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor):
        """
        x: [B,T,100]
        We take the representation at the final timestep (most recent draw in context)
        to predict the next draw.
        """
        B, T, D = x.shape
        assert T == self.context_len and D == 100

        h = self.in_proj(x)
        pos = torch.arange(T, device=x.device).unsqueeze(0)  # [1,T]
        h = h + self.pos_emb(pos)

        h = self.enc(h, mask=self.causal_mask)
        h = self.ln(h)

        last = h[:, -1, :]  # [B,d_model]
        logits_main = self.head_main(last)   # [B,50]
        logits_bonus = self.head_bonus(last) # [B,50]
        return logits_main, logits_bonus


# ---------------------------
# Metrics
# ---------------------------

@torch.no_grad()
def topk_hits(pred_logits: torch.Tensor, target_multi_hot: torch.Tensor, k: int = 7) -> torch.Tensor:
    """
    Returns hits count per example: number of correct items in top-k.
    pred_logits: [B,50]
    target_multi_hot: [B,50] {0,1}
    """
    topk = torch.topk(pred_logits, k=k, dim=-1).indices  # [B,k]
    hits = target_multi_hot.gather(1, topk).sum(dim=1)   # [B]
    return hits

def compute_loss_and_metrics(logits_main, logits_bonus, target_main, target_bonus):
    """
    logits_main: [B,50]
    logits_bonus: [B,50]
    target_main: [B,50] float {0,1}
    target_bonus: [B] long
    """
    # Main: multi-label BCE
    loss_main = F.binary_cross_entropy_with_logits(logits_main, target_main)

    # Bonus: mask out the 7 main numbers (bonus can't be any of them)
    masked_bonus = logits_bonus.clone()
    masked_bonus[target_main.bool()] = -1e9
    loss_bonus = F.cross_entropy(masked_bonus, target_bonus)

    loss = loss_main + loss_bonus

    # Metrics
    hits7 = topk_hits(logits_main, target_main, k=7)  # [B]
    hit_rate = (hits7 / 7.0).mean()                   # avg fraction correct among the 7
    exact7 = (hits7 == 7).float().mean()

    pred_bonus = masked_bonus.argmax(dim=-1)
    bonus_acc = (pred_bonus == target_bonus).float().mean()

    return loss, loss_main.detach(), loss_bonus.detach(), hit_rate.detach(), exact7.detach(), bonus_acc.detach()


@torch.no_grad()
def evaluate(model, loader, device, max_batches: int = 200):
    model.eval()
    agg = {
        "loss": [],
        "loss_main": [],
        "loss_bonus": [],
        "hit_rate": [],
        "exact7": [],
        "bonus_acc": [],
    }
    for i, (x, y_main, y_bonus) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y_main = y_main.to(device, non_blocking=True)
        y_bonus = y_bonus.to(device, non_blocking=True)

        lm, lb = model(x)
        loss, lm_loss, lb_loss, hit_rate, exact7, bonus_acc = compute_loss_and_metrics(lm, lb, y_main, y_bonus)
        agg["loss"].append(loss)
        agg["loss_main"].append(lm_loss)
        agg["loss_bonus"].append(lb_loss)
        agg["hit_rate"].append(hit_rate)
        agg["exact7"].append(exact7)
        agg["bonus_acc"].append(bonus_acc)

    out = {}
    for k, vals in agg.items():
        if not vals:
            out[k] = torch.tensor(float("nan"), device=device)
        else:
            out[k] = torch.stack(vals).mean()
            out[k] = all_reduce_mean(out[k])
    return out


# ---------------------------
# Training
# ---------------------------

@dataclass
class Config:
    context_len: int = 64          # number of past draws to condition on
    batch_size: int = 256
    epochs: int = 5
    lr: float = 3e-4
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    d_model: int = 512
    nhead: int = 8
    num_layers: int = 6
    dropout: float = 0.1
    amp: bool = True
    log_every: int = 50
    eval_every: int = 250
    num_workers: int = 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--context_len", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--train_split", type=float, default=0.9)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--ckpt", type=str, default="checkpoints/lottomax_set.pt")
    args = p.parse_args()

    # DDP init (torchrun sets env vars)
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        torch.distributed.init_process_group(backend=("nccl" if torch.cuda.is_available() else "gloo"))

    seed_all(args.seed + ddp_rank())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    cfg = Config(
        context_len=args.context_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        amp=not args.no_amp,
        log_every=args.log_every,
        eval_every=args.eval_every,
        num_workers=args.num_workers,
    )

    draws = load_lottomax_csv(args.data_path)
    log_rank0(f"Loaded {len(draws)} draws from {args.data_path}")

    split = int(len(draws) * args.train_split)
    train_draws = draws[:split]
    val_draws = draws[split - cfg.context_len:]  # overlap

    train_ds = NextDrawDataset(train_draws, cfg.context_len)
    val_ds = NextDrawDataset(val_draws, cfg.context_len)

    train_sampler = DistributedSampler(train_ds, shuffle=True) if ddp_is_enabled() else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if ddp_is_enabled() else None

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=min(cfg.batch_size, 16),
        sampler=val_sampler,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = DrawTransformer(
        context_len=cfg.context_len,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)

    if ddp_is_enabled():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    # Chance baselines:
    # - Main set: expected overlap when you pick 7 uniformly from 50 is 7*(7/50)=0.98 hits => hit_rate ~ 0.14
    expected_hits = 7.0 * (7.0 / 50.0)
    expected_hit_rate = expected_hits / 7.0
    # - Bonus: 1/(50-7) chance if you mask the 7 mains
    expected_bonus_acc = 1.0 / 43.0
    log_rank0(f"Chance-ish baselines: main_hit_rate≈{expected_hit_rate:.4f}, bonus_top1_acc≈{expected_bonus_acc:.4f}")

    global_step = 0
    t0 = time.time()

    def save_ckpt(step: int):
        if ddp_rank() != 0:
            return
        os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)
        core = model.module if hasattr(model, "module") else model
        torch.save({"model": core.state_dict(), "step": step, "cfg": cfg.__dict__}, args.ckpt)
        log_rank0(f"Saved checkpoint: {args.ckpt}")

    for epoch in range(cfg.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        for x, y_main, y_bonus in train_loader:
            global_step += 1
            x = x.to(device, non_blocking=True)
            y_main = y_main.to(device, non_blocking=True)
            y_bonus = y_bonus.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(cfg.amp and device.type == "cuda")):
                lm, lb = model(x)
                loss, loss_main, loss_bonus, hit_rate, exact7, bonus_acc = compute_loss_and_metrics(lm, lb, y_main, y_bonus)

            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)

            if global_step % cfg.log_every == 0:
                loss_r = all_reduce_mean(loss.detach())
                hit_r = all_reduce_mean(hit_rate)
                bacc_r = all_reduce_mean(bonus_acc)

                elapsed = time.time() - t0
                examples = cfg.batch_size * cfg.log_every * ddp_world_size()
                ex_per_s = examples / max(elapsed, 1e-6)
                t0 = time.time()

                log_rank0(
                    f"[epoch {epoch+1}/{cfg.epochs} step {global_step}] "
                    f"train_loss={loss_r.item():.4f} "
                    f"main_hit_rate={hit_r.item():.4f} "
                    f"bonus_acc={bacc_r.item():.4f} "
                    f"ex/s={ex_per_s:,.0f}"
                )

            if global_step % cfg.eval_every == 0:

                ddp_barrier()
                m = evaluate(model, val_loader, device)
                log_rank0(
                    f"  eval: loss={m['loss'].item():.4f} "
                    f"main_hit_rate={m['hit_rate'].item():.4f} "
                    f"main_exact7={m['exact7'].item():.6f} "
                    f"bonus_acc={m['bonus_acc'].item():.4f}"
                )
                save_ckpt(global_step)
                ddp_barrier()

    ddp_barrier()
    m = evaluate(model, val_loader, device, max_batches=500)
    log_rank0(
        f"FINAL eval: loss={m['loss'].item():.4f} "
        f"main_hit_rate={m['hit_rate'].item():.4f} "
        f"main_exact7={m['exact7'].item():.6f} "
        f"bonus_acc={m['bonus_acc'].item():.4f}"
    )
    save_ckpt(global_step)

    if ddp_is_enabled():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()



