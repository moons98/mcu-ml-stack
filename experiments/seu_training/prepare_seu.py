"""Preprocess the SEU gearbox/bearing dataset (reproducing Gao et al., Sci China Tech Sci 2025).

Settings from the paper (Sec. 2.2 + Supporting Information Tables S2/S3):
- 20 Hz speed condition (*_20_0.csv), 4 classes: health / inner / ball / outer
- 96000 timesteps per class -> 6000 non-overlapping windows of 16 timesteps
- train/test/val = 4000/1000/1000 per class (4:1:1)
- Input (16, 6), MinMaxScaler normalization

Assumptions (not specified in the paper):
- 6 channels = ch2-4 (planetary gearbox x,y,z) + ch6-8 (parallel gearbox x,y,z)  (see --channels)
- The first 96000 rows of each file are used
- MinMaxScaler is fit per channel on the train split only
- Split: random (presumed paper setting) / chrono (time-ordered, to check for leakage)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ["health", "inner", "ball", "outer"]  # label 0..3
HEADER_ROWS = 16  # metadata rows up to and including the 'Data' line


def load_csv(path: Path, n_rows: int) -> np.ndarray:
    # ball_20_0.csv is comma-separated, the others are tab-separated: detect the delimiter
    with open(path, "r", errors="replace") as f:
        first = f.readline()
    sep = "," if first.count(",") > first.count("\t") else "\t"
    df = pd.read_csv(path, sep=sep, skiprows=HEADER_ROWS, header=None,
                     nrows=n_rows, usecols=range(8), dtype=np.float32)
    arr = df.to_numpy()
    if arr.shape != (n_rows, 8) or not np.isfinite(arr).all():
        raise ValueError(f"{path.name}: unexpected shape {arr.shape} or non-finite values")
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path(__file__).resolve().parents[2] / "data/seu/raw")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[2] / "data/seu/processed")
    ap.add_argument("--channels", type=str, default="2,3,4,6,7,8", help="1-based channel numbers")
    ap.add_argument("--rows", type=int, default=96000)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--split", choices=["random", "chrono"], default="random")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ch_idx = [int(c) - 1 for c in args.channels.split(",")]
    n_win = args.rows // args.window
    n_tr, n_te = n_win * 4 // 6, n_win // 6
    rng = np.random.default_rng(args.seed)

    splits = {k: ([], []) for k in ("train", "val", "test")}
    for label, name in enumerate(CLASSES):
        raw = load_csv(args.raw / f"{name}_20_0.csv", args.rows)[:, ch_idx]
        win = raw[: n_win * args.window].reshape(n_win, args.window, len(ch_idx))
        idx = rng.permutation(n_win) if args.split == "random" else np.arange(n_win)
        # Table S2 order: train 4000 / test 1000 / val 1000
        parts = {"train": idx[:n_tr], "test": idx[n_tr:n_tr + n_te], "val": idx[n_tr + n_te:]}
        for k, ii in parts.items():
            splits[k][0].append(win[ii])
            splits[k][1].append(np.full(len(ii), label, dtype=np.int64))
        print(f"{name:7s} mean per ch: {np.round(raw.mean(0), 4)}  std: {np.round(raw.std(0), 4)}")

    data = {k: (np.concatenate(x), np.concatenate(y)) for k, (x, y) in splits.items()}

    # MinMaxScaler (per channel, fit on train) -> [0, 1]
    x_tr = data["train"][0]
    lo = x_tr.reshape(-1, x_tr.shape[-1]).min(0)
    hi = x_tr.reshape(-1, x_tr.shape[-1]).max(0)
    scale = np.where(hi > lo, hi - lo, 1.0)

    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.split}"
    out = {}
    for k, (x, y) in data.items():
        out[f"x_{k}"] = ((x - lo) / scale).astype(np.float32)
        out[f"y_{k}"] = y
        print(f"{k:5s} x={out[f'x_{k}'].shape} y={np.bincount(y)}")
    np.savez_compressed(args.out / f"seu_{tag}.npz", **out)
    meta = {"channels_1based": args.channels, "rows": args.rows, "window": args.window,
            "split": args.split, "seed": args.seed, "classes": CLASSES,
            "minmax_lo": lo.tolist(), "minmax_hi": hi.tolist()}
    (args.out / f"seu_{tag}.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {args.out / f'seu_{tag}.npz'}")


if __name__ == "__main__":
    main()
