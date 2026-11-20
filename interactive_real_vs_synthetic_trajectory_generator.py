"""
===============================================================================
Interactive Conditional Battery Trajectory Generator
===============================================================================

Authors:
    Volkan Yamaçlı, Timur Lale, Serdar Ekinci, Rıdvan Fırat Çınar,
    Davut İzci, Gökhan Yüksek

Article:
    Conditional Diffusion for Probabilistic and Controllable Synthetic
    Battery Charge-Discharge Curves

Purpose:
    Select a dataset (CALCE, NASA or Severson), a battery, a step type
    (charge/discharge) and 1-10 trajectories. One real segment of the selected
    battery and step type is chosen (at random with a fixed seed, or by its
    life index). Its protocol-setting condition vector is built, the
    dataset's best_model_ema.pt generates synthetic voltage and local-capacity
    trajectories for that condition (ancestral sampling, 500 steps, cosine
    schedule), the capacity channel is post-processed, the curves are mapped
    to physical units (V, Ah) and plotted against the real segment.
    Mean +/-1 SD voltage and capacity figures are saved as PNG, SVG and PDF.

Expected layout:
    script.py
    calce/    metadata.csv, samples/, best_model_ema.pt, global_stats.json, life_scale.json
    nasa/     metadata.csv, samples/, best_model_ema.pt, global_stats.json, life_scale.json
    severson/ metadata.csv, samples/, best_model_ema.pt, global_stats.json, life_scale.json

Usage:
    python interactive_real_vs_synthetic_trajectory_generator.py            (interactive)
    python interactive_real_vs_synthetic_trajectory_generator.py --list
    python interactive_real_vs_synthetic_trajectory_generator.py \
        --dataset nasa --battery B0006 --step discharge --count 5
===============================================================================
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import savgol_filter
from sklearn.isotonic import IsotonicRegression

BASE_DIR = Path(__file__).resolve().parent
DATASETS = ("calce", "nasa", "severson")
DATASET_LABELS = {"calce": "CALCE", "nasa": "NASA", "severson": "Severson"}
OUTPUT_DIR = BASE_DIR / "generated_trajectories"
SEQ_LEN = 512
CHANNELS = 2                     # 0: voltage [V], 1: local capacity [Ah]
TIMESTEPS = 500
BASE, EMB_DIM, NUM_HEADS = 96, 384, 8
MAX_TRAJECTORIES = 10
SEED = 42


# =============================================================================
# NETWORK
# =============================================================================

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.emb_proj = nn.Linear(emb_dim, out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x, emb):
        h = self.act(self.norm1(self.conv1(x)))
        h = h + self.emb_proj(emb)[:, :, None]
        h = self.act(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class SelfAttention1D(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        h = self.norm(x).transpose(1, 2)
        h, _ = self.attn(h, h, h, need_weights=False)
        return x + h.transpose(1, 2)


class ConditionalUNet1D(nn.Module):
    def __init__(self, channels=2, cond_dim=4, base=96, emb_dim=384, num_heads=8):
        super().__init__()
        self.time_emb = nn.Sequential(
            SinusoidalTimeEmbedding(emb_dim),
            nn.Linear(emb_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim),
        )
        self.cond_emb = nn.Sequential(
            nn.Linear(cond_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim),
        )
        self.in_conv = nn.Conv1d(channels, base, 3, padding=1)

        self.down1 = ResBlock1D(base, base, emb_dim)
        self.downsample1 = nn.Conv1d(base, base * 2, 4, stride=2, padding=1)
        self.down2 = ResBlock1D(base * 2, base * 2, emb_dim)
        self.downsample2 = nn.Conv1d(base * 2, base * 4, 4, stride=2, padding=1)
        self.down3 = ResBlock1D(base * 4, base * 4, emb_dim)
        self.downsample3 = nn.Conv1d(base * 4, base * 8, 4, stride=2, padding=1)

        self.mid1 = ResBlock1D(base * 8, base * 8, emb_dim)
        self.attn = SelfAttention1D(base * 8, num_heads=num_heads)
        self.mid2 = ResBlock1D(base * 8, base * 8, emb_dim)

        self.upsample3 = nn.ConvTranspose1d(base * 8, base * 4, 4, stride=2, padding=1)
        self.up3 = ResBlock1D(base * 8, base * 4, emb_dim)
        self.upsample2 = nn.ConvTranspose1d(base * 4, base * 2, 4, stride=2, padding=1)
        self.up2 = ResBlock1D(base * 4, base * 2, emb_dim)
        self.upsample1 = nn.ConvTranspose1d(base * 2, base, 4, stride=2, padding=1)
        self.up1 = ResBlock1D(base * 2, base, emb_dim)

        self.out = nn.Sequential(nn.GroupNorm(8, base), nn.SiLU(), nn.Conv1d(base, channels, 3, padding=1))

    def forward(self, x, t, cond):
        emb = self.time_emb(t) + self.cond_emb(cond)
        x0 = self.in_conv(x)
        d1 = self.down1(x0, emb)
        d2 = self.down2(self.downsample1(d1), emb)
        d3 = self.down3(self.downsample2(d2), emb)
        x = self.mid2(self.attn(self.mid1(self.downsample3(d3), emb)), emb)
        x = self.up3(torch.cat([self.upsample3(x), d3], dim=1), emb)
        x = self.up2(torch.cat([self.upsample2(x), d2], dim=1), emb)
        x = self.up1(torch.cat([self.upsample1(x), d1], dim=1), emb)
        return self.out(x)


# =============================================================================
# DIFFUSION (cosine schedule, ancestral sampling)
# =============================================================================

class CosineDiffusion:
    def __init__(self, timesteps=500, device="cpu", s=0.008):
        self.timesteps = timesteps
        self.device = device
        x = torch.linspace(0, timesteps, timesteps + 1, device=device)
        alpha_bar = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        beta = torch.clip(1 - (alpha_bar[1:] / alpha_bar[:-1]), 1e-4, 0.999)
        self.beta = beta
        self.alpha = 1.0 - beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    @torch.no_grad()
    def sample(self, model, shape, cond, generator=None, clamp=3.0):
        model.eval()
        x = torch.randn(shape, device=self.device, generator=generator)
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=self.device).float()
            pred_noise = model(x, t, cond)
            alpha_t, alpha_bar_t, beta_t = self.alpha[i], self.alpha_bar[i], self.beta[i]
            noise = (torch.randn(shape, device=self.device, generator=generator)
                     if i > 0 else torch.zeros_like(x))
            x = (1 / torch.sqrt(alpha_t)
                 * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * pred_noise)
                 + torch.sqrt(beta_t) * noise)
            x = torch.clamp(x, -clamp, clamp)
        return x


# =============================================================================
# CONDITION VECTOR (protocol settings only; no battery identity)
# =============================================================================
#   CALCE    [life, current, step]                                    (cond_dim 3)
#   NASA     [life_abs, current_nom, step, temperature, cutoff, load_sq]  (cond_dim 6)
#   Severson [life_abs, step, C1, Q1, C2]                             (cond_dim 5)

COND_COLUMNS = {
    "calce": ["life", "current", "step"],
    "nasa": ["life_abs", "current_nom", "step", "temperature", "cutoff", "load_sq"],
    "severson": ["life_abs", "step", "c1", "q1", "c2"],
}
NPZ_COND_4 = ["battery", "life", "current", "step"]          # CALCE / NASA .npz 'cond'
SEV_RAW_COND = ["life_index", "step_code", "C1", "Q1", "C2"]  # Severson .npz 'cond'
SEV_CRATE_SCALE = 8.0
SEV_Q1_SCALE = 100.0
SEV_COLUMNS = {"c1": ("C1", SEV_CRATE_SCALE), "q1": ("Q1", SEV_Q1_SCALE), "c2": ("C2", SEV_CRATE_SCALE)}
TEMP_BOUNDS = (4.0, 44.0)
CUTOFF_BOUNDS = (2.0, 2.7)
CURRENT_NOM_BOUNDS = (0.0, 4.0)
# NASA test-protocol settings per battery (discharge cut-off voltage in V, square-wave load)
NASA_CUTOFF_V = {
    "B0005": 2.7, "B0006": 2.5, "B0007": 2.2, "B0018": 2.5,
    "B0025": 2.0, "B0026": 2.2, "B0027": 2.5, "B0028": 2.7,
    "B0029": 2.0, "B0030": 2.2, "B0031": 2.5, "B0032": 2.7,
    "B0033": 2.0, "B0034": 2.2, "B0036": 2.7,
    "B0038": 2.2, "B0039": 2.5, "B0040": 2.7,
    "B0041": 2.0, "B0042": 2.2, "B0043": 2.5, "B0044": 2.7,
    "B0045": 2.0, "B0046": 2.2, "B0047": 2.5, "B0048": 2.7,
    "B0049": 2.0, "B0050": 2.2, "B0051": 2.5, "B0052": 2.7,
    "B0053": 2.0, "B0054": 2.2, "B0055": 2.5, "B0056": 2.7,
}
NASA_CHARGE_CC_A = 1.5
NASA_DISCHARGE_LEVELS_A = (1.0, 2.0, 4.0)
NASA_LOAD_SQ = {b: (1 if b in ("B0025", "B0026", "B0027", "B0028") else 0) for b in NASA_CUTOFF_V}
NASA_SQ_DUTY = 0.5


def nasa_nominal_current(meta):
    """Nominal time-averaged set-point current |A| per NASA segment."""
    levels = np.asarray(NASA_DISCHARGE_LEVELS_A, dtype=np.float64)
    load = -meta["current_min"].to_numpy(np.float64)
    level = levels[np.abs(load[:, None] - levels[None, :]).argmin(1)]
    sq = meta["battery_id"].map(NASA_LOAD_SQ).to_numpy(np.float64)
    dis = (meta["step_type"] == "discharge").to_numpy()
    return np.where(dis, level * np.where(sq == 1, NASA_SQ_DUTY, 1.0), NASA_CHARGE_CC_A)


def build_condition(dataset, cond, meta, life_scale):
    """cond: [N, 4] (CALCE/NASA) or [N, 5] (Severson) .npz vectors; meta: matching metadata rows."""
    sev = dataset == "severson"
    cols = []
    for name in COND_COLUMNS[dataset]:
        if name in SEV_COLUMNS:
            raw, scale = SEV_COLUMNS[name]
            cols.append((cond[:, SEV_RAW_COND.index(raw)].astype(np.float64) / scale).astype(np.float32))
        elif sev and name == "step":
            cols.append(cond[:, SEV_RAW_COND.index("step_code")].astype(np.float32))
        elif name == "life_abs":
            cols.append((meta["life_index"].to_numpy(np.float64) / life_scale).astype(np.float32))
        elif name == "current_nom":
            amps = nasa_nominal_current(meta)
            cols.append(((amps - CURRENT_NOM_BOUNDS[0]) / (CURRENT_NOM_BOUNDS[1] - CURRENT_NOM_BOUNDS[0])).astype(np.float32))
        elif name == "load_sq":
            cols.append(meta["battery_id"].map(NASA_LOAD_SQ).to_numpy(np.float32))
        elif name == "temperature":
            t = meta["ambient_temperature"].to_numpy(np.float32)
            cols.append((t - TEMP_BOUNDS[0]) / (TEMP_BOUNDS[1] - TEMP_BOUNDS[0]))
        elif name == "cutoff":
            v = meta["battery_id"].map(NASA_CUTOFF_V).to_numpy(np.float32)
            cols.append((v - CUTOFF_BOUNDS[0]) / (CUTOFF_BOUNDS[1] - CUTOFF_BOUNDS[0]))
        else:
            cols.append(cond[:, NPZ_COND_4.index(name)])
    return np.stack(cols, axis=1).astype(np.float32)


# =============================================================================
# CAPACITY POST-PROCESSING AND PHYSICAL UNITS
# =============================================================================

SG_WINDOW, SG_ORDER, CAP_CH = 31, 2, 1


def postprocess(z):
    """z [N, 2, L] -> float64 [N, 2, L]. Both channels clipped to [0, 1]; on the capacity channel
    only: Savitzky-Golay (window 31, order 2, mode "interp"), clip to [0, 1], then isotonic
    (non-decreasing) least-squares projection. The voltage channel is otherwise unchanged."""
    b = np.clip(z, 0, 1).astype(np.float64)
    q = b.copy()
    q[:, CAP_CH] = np.clip(savgol_filter(b[:, CAP_CH], SG_WINDOW, SG_ORDER, axis=-1, mode="interp"), 0, 1)
    y = q.copy()
    ir = IsotonicRegression(increasing=True)
    t = np.arange(q.shape[-1])
    for n in range(len(q)):
        y[n, CAP_CH] = ir.fit_transform(t, q[n, CAP_CH])
    return y


def denormalize(x, stats):
    """Globally normalised arrays [..., 2, L] -> physical units (V, Ah)."""
    mn = np.asarray(stats["min"], dtype=np.float32)[:, None]
    mx = np.asarray(stats["max"], dtype=np.float32)[:, None]
    return x * (mx - mn) + mn


# =============================================================================
# LOADING AND GENERATION
# =============================================================================

def dataset_paths(dataset):
    root = BASE_DIR / dataset
    return dict(root=root, metadata=root / "metadata.csv", samples=root / "samples",
                model=root / "best_model_ema.pt", stats=root / "global_stats.json",
                life=root / "life_scale.json")


def load_dataset(dataset):
    p = dataset_paths(dataset)
    for key in ("metadata", "samples", "model", "stats", "life"):
        if not p[key].exists():
            raise FileNotFoundError(f"Required path was not found: {p[key]}")
    dtype = {"segment_flag": str, "test_set": str} if dataset == "severson" else None
    metadata = pd.read_csv(p["metadata"], dtype=dtype)
    metadata["battery_id"] = metadata["battery_id"].astype(str).str.strip()
    metadata["step_type"] = metadata["step_type"].astype(str).str.lower().str.strip()
    stats = json.loads(p["stats"].read_text(encoding="utf-8"))
    life_scale = json.loads(p["life"].read_text(encoding="utf-8"))["life_scale"]
    return metadata, stats, life_scale


def load_model(dataset, device):
    model = ConditionalUNet1D(CHANNELS, len(COND_COLUMNS[dataset]), BASE, EMB_DIM, NUM_HEADS).to(device)
    state = torch.load(dataset_paths(dataset)["model"], map_location=device, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def select_segment(metadata, battery, step, life=None, seed=SEED):
    rows = metadata[(metadata["battery_id"] == battery) & (metadata["step_type"] == step)]
    if rows.empty:
        raise ValueError(f"No {step} samples exist for {battery}.")
    if life is not None:
        rows = rows[rows["life_index"] == int(life)]
        if rows.empty:
            raise ValueError(f"No {step} sample of {battery} has life index {life}.")
        return rows.iloc[[0]]
    rng = np.random.default_rng(seed)
    return rows.iloc[[int(rng.integers(len(rows)))]]


def segment_condition(dataset, row, life_scale):
    """Condition vector [1, cond_dim] and real physical curve [2, 512] of one metadata row."""
    with np.load(dataset_paths(dataset)["samples"] / str(row["file"].iloc[0])) as data:
        cond = data["cond"].astype(np.float32)[None]
        real = data["x_raw"].astype(np.float32)
    return build_condition(dataset, cond, row.reset_index(drop=True), life_scale), real


def generate(dataset, condition, count, seed=SEED, device="cpu", model=None):
    """count trajectories for one condition [1, cond_dim]. Returns (raw model output [count, 2, 512]
    in normalised units, post-processed output [count, 2, 512] in normalised units)."""
    device = torch.device(device)
    model = model if model is not None else load_model(dataset, device)
    diffusion = CosineDiffusion(TIMESTEPS, device=device)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    c = torch.from_numpy(np.repeat(condition, count, axis=0)).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        x = diffusion.sample(model, (count, CHANNELS, SEQ_LEN), c, generator=generator)
    raw = x.float().cpu().numpy()
    return raw, postprocess(raw)


# =============================================================================
# FIGURES
# =============================================================================

def save_figure(fig, stem):
    import matplotlib.pyplot as plt
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), format="svg", bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), format="pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_distribution(samples, real_signal, dataset, battery, step, life, output_dir):
    """Real trajectory with the individual synthetic trajectories, their mean and the +/-1 SD band."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="paper", font="Times New Roman", font_scale=1.25)
    palette = sns.color_palette("pastel", 4)
    x = np.arange(samples.shape[-1])
    settings = [
        (0, "Voltage (V)", "voltage", palette[0]),
        (1, "Local Capacity (Ah)", "capacity", palette[1]),
    ]
    stems = []
    for channel, ylabel, suffix, color in settings:
        mean = samples[:, channel].mean(axis=0)
        std = samples[:, channel].std(axis=0)
        fig, ax = plt.subplots(figsize=(9.5, 5.6))
        for index, sample in enumerate(samples):
            ax.plot(x, sample[channel], color=color, alpha=0.18, linewidth=1.0,
                    label="Synthetic Samples" if index == 0 else None, zorder=1)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.30, label="Synthetic ±1 SD", zorder=2)
        ax.plot(x, mean, color=color, linewidth=2.8, label="Synthetic Mean", zorder=3)
        ax.plot(x, real_signal[channel], color="black", linewidth=2.2, linestyle="--",
                label="Real Trajectory", zorder=4)
        ax.set_xlabel("Normalized Time Index")
        ax.set_ylabel(ylabel)
        ax.legend(title=f"{DATASET_LABELS[dataset]} | {battery} | {step.capitalize()} | Life {life}",
                  fontsize=9, title_fontsize=9)
        sns.despine(ax=ax)
        fig.tight_layout()
        stem = output_dir / f"{dataset}_{battery}_{step}_life{life}_{suffix}"
        save_figure(fig, stem)
        stems.append(stem)
    return stems


# =============================================================================
# USER INTERFACE
# =============================================================================

def ask_choice(prompt, choices):
    while True:
        value = input(prompt).strip()
        lowered = {c.lower(): c for c in choices}
        if value.lower() in lowered:
            return lowered[value.lower()]
        print("Invalid selection. Available choices: " + ", ".join(choices))


def ask_int(prompt, low, high):
    while True:
        try:
            value = int(input(prompt).strip())
        except ValueError:
            print("Please enter a whole number.")
            continue
        if low <= value <= high:
            return value
        print(f"Please enter a number between {low} and {high}.")


def choose_battery(metadata):
    batteries = sorted(metadata["battery_id"].unique())
    print("\nAvailable Batteries")
    print("-" * 40)
    for index, battery in enumerate(batteries, 1):
        print(f"{index:>3}. {battery}")
    return batteries[ask_int("\nSelect a battery number: ", 1, len(batteries)) - 1]


def print_overview():
    print("\n" + "=" * 72)
    print("INTERACTIVE CONDITIONAL BATTERY TRAJECTORY GENERATOR")
    print("=" * 72)
    print("Generates synthetic voltage and local-capacity trajectories with the")
    print("trained CDDPM of each dataset for the protocol settings of a real segment")
    print("and plots them (mean, +/-1 SD, individual samples) against that segment.")
    print(f"Sampling: {TIMESTEPS} ancestral steps, cosine schedule; default seed {SEED}.")
    for dataset in DATASETS:
        p = dataset_paths(dataset)
        status = "ready" if all(p[k].exists() for k in ("metadata", "samples", "model", "stats", "life")) else "missing files"
        print(f"  {DATASET_LABELS[dataset]:<9} condition [{', '.join(COND_COLUMNS[dataset])}]  ({status})")


def list_contents():
    print_overview()
    for dataset in DATASETS:
        metadata, _, _ = load_dataset(dataset)
        counts = metadata.groupby(["battery_id", "step_type"]).size().unstack(fill_value=0)
        print(f"\n{DATASET_LABELS[dataset]}: {len(metadata)} segments, {len(counts)} batteries")
        print("  " + ", ".join(f"{b} ({int(r.get('charge', 0))}/{int(r.get('discharge', 0))})"
                               for b, r in counts.iterrows()))
        print("  (charge/discharge segments per battery)")


def run(dataset, battery, step, count, life=None, seed=SEED, output_dir=OUTPUT_DIR, device=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata, stats, life_scale = load_dataset(dataset)
    row = select_segment(metadata, battery, step, life, seed)
    condition, real = segment_condition(dataset, row, life_scale)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    life_index = int(row["life_index"].iloc[0])
    print(f"\nDevice: {device}")
    print(f"Segment: {DATASET_LABELS[dataset]} | {battery} | {step.capitalize()} | Life {life_index}"
          f" | {row['file'].iloc[0]}")
    print("Condition: " + ", ".join(f"{n}={v:.4f}" for n, v in zip(COND_COLUMNS[dataset], condition[0])))
    _, generated = generate(dataset, condition, count, seed=seed, device=device)
    physical = denormalize(generated.astype(np.float32), stats)
    stems = plot_distribution(physical, real, dataset, battery, step, life_index, output_dir)
    print(f"\nGeneration completed. Output directory: {output_dir}")
    for stem in stems:
        print(f"  - {stem.name}.png/.svg/.pdf")
    return physical, real


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic battery trajectories for the condition of a real segment.")
    parser.add_argument("--list", action="store_true", help="list datasets, batteries and segment counts, then exit")
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--battery")
    parser.add_argument("--step", choices=["charge", "discharge"])
    parser.add_argument("--count", type=int, help=f"number of trajectories (1-{MAX_TRAJECTORIES})")
    parser.add_argument("--life", type=int, help="life index of the real segment (default: random with --seed)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR, help="output directory")
    args = parser.parse_args()
    matplotlib.use("Agg")

    if args.list:
        list_contents()
        return
    print_overview()
    dataset = args.dataset or ask_choice("\nSelect a dataset (calce/nasa/severson): ", list(DATASETS))
    metadata, _, _ = load_dataset(dataset)
    battery = args.battery or choose_battery(metadata)
    if battery not in set(metadata["battery_id"]):
        raise ValueError(f"Battery {battery} is not in the {DATASET_LABELS[dataset]} metadata.")
    steps = sorted(metadata.loc[metadata["battery_id"] == battery, "step_type"].unique())
    step = args.step or ask_choice(f"\nSelect a step type ({'/'.join(steps)}): ", steps)
    count = args.count or ask_int(f"\nHow many trajectories should be generated? (1-{MAX_TRAJECTORIES}): ",
                                  1, MAX_TRAJECTORIES)
    if not 1 <= count <= MAX_TRAJECTORIES:
        raise ValueError(f"--count must be between 1 and {MAX_TRAJECTORIES}.")
    run(dataset, battery, step, count, life=args.life, seed=args.seed, output_dir=args.out, device=args.device)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by the user.")
    except Exception as error:
        print("\nThe program could not complete the operation.")
        print(f"Reason: {error}")
        raise SystemExit(1)
