"""
===============================================================================
Interactive Battery Curves-over-Life Visualizer
===============================================================================

Authors:
    Volkan Yamaçlı, Timur Lale, Serdar Ekinci, Rıdvan Fırat Çınar,
    Davut İzci, Gökhan Yüksek

Article:
    Conditional Diffusion for Probabilistic and Controllable Synthetic
    Battery Charge-Discharge Curves

Purpose:
    Select a dataset (CALCE, NASA or Severson), a battery and a step type.
    The real charge or discharge segments of that battery, from early to late
    life, are drawn on one axis with a colour gradient by cycle number and a
    colour bar, in physical units: voltage against local capacity, and
    voltage and local capacity against the sample index. Real data only; no
    model is loaded. Saved as PNG, SVG and PDF.

Outputs:
    <dataset>_<battery>_<step>_life_evolution.png / .svg / .pdf
    <dataset>_<battery>_<step>_life_evolution_records.csv

Usage:
    python interactive_life_evolution_visualizer.py                    (interactive)
    python interactive_life_evolution_visualizer.py --list
    python interactive_life_evolution_visualizer.py --dataset nasa --battery B0005 --step discharge
===============================================================================
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd


# =============================================================================
# PATHS AND SETTINGS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
DATASETS = ("calce", "nasa", "severson")
DATASET_LABELS = {"calce": "CALCE", "nasa": "NASA", "severson": "Severson"}
OUTPUT_DIR = BASE_DIR / "output"

MAX_CURVES = 60                  # default number of segments drawn (evenly spaced over life)
RANDOM_SEED = 42                 # used with --sampling random
PNG_DPI = 300
EXPECTED_CHANNELS = 2
COLORMAP = "viridis"


# =============================================================================
# STYLE
# =============================================================================

def configure_plot_style() -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="paper", font="Times New Roman", font_scale=1.25)
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": PNG_DPI,
        "axes.linewidth": 1.1,
        "axes.edgecolor": "0.2",
        "grid.linewidth": 0.55,
        "grid.alpha": 0.30,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


# =============================================================================
# USER INPUT
# =============================================================================

def ask_choice(prompt: str, choices: list[str]) -> str:
    while True:
        value = input(prompt).strip()
        lowered = {c.lower(): c for c in choices}
        if value.lower() in lowered:
            return lowered[value.lower()]
        print("Invalid selection. Available choices: " + ", ".join(choices))


def ask_int(prompt: str, low: int, high: int, default: int | None = None) -> int:
    while True:
        raw = input(prompt).strip()
        if not raw and default is not None:
            return default
        try:
            value = int(raw)
        except ValueError:
            print("Please enter a whole number.")
            continue
        if low <= value <= high:
            return value
        print(f"Please enter a number between {low} and {high}.")


def choose_battery(metadata: pd.DataFrame) -> str:
    batteries = sorted(metadata["battery_id"].unique())
    print("\nAvailable Batteries")
    print("-" * 40)
    for index, battery in enumerate(batteries, 1):
        print(f"{index:>3}. {battery}")
    return batteries[ask_int("\nSelect a battery number: ", 1, len(batteries)) - 1]


# =============================================================================
# DATA LOADING
# =============================================================================

def processed_dir(dataset: str) -> Path:
    return BASE_DIR / dataset


def load_metadata(dataset: str) -> pd.DataFrame:
    root = processed_dir(dataset)
    missing = [p for p in (root, root / "metadata.csv", root / "samples") if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{DATASET_LABELS[dataset]} processed dataset is incomplete.\nMissing paths:\n"
                                + "\n".join(f"  - {p}" for p in missing))
    dtype = {"segment_flag": str, "test_set": str} if dataset == "severson" else None
    metadata = pd.read_csv(root / "metadata.csv", dtype=dtype)
    missing_columns = {"battery_id", "step_type", "life_index", "file"} - set(metadata.columns)
    if missing_columns:
        raise KeyError(f"{DATASET_LABELS[dataset]} metadata.csv is missing required columns: "
                       f"{sorted(missing_columns)}")
    metadata["battery_id"] = metadata["battery_id"].astype(str).str.strip()
    metadata["step_type"] = metadata["step_type"].astype(str).str.lower().str.strip()
    return metadata[metadata["step_type"].isin(["charge", "discharge"])].copy()


def load_physical_signal(dataset: str, file_name: str) -> np.ndarray:
    sample_path = processed_dir(dataset) / "samples" / file_name
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample file was not found: {sample_path}")
    with np.load(sample_path) as data:
        signal = data["x_raw"].astype(np.float32)
    if signal.ndim != 2 or signal.shape[0] != EXPECTED_CHANNELS:
        raise ValueError(f"Unexpected x_raw shape in {sample_path}: {signal.shape}")
    return signal


def select_rows(metadata: pd.DataFrame, battery: str, step: str, max_curves: int,
                sampling: str, seed: int) -> pd.DataFrame:
    rows = (metadata[(metadata["battery_id"] == battery) & (metadata["step_type"] == step)]
            .sort_values("life_index").reset_index(drop=True))
    if rows.empty:
        raise ValueError(f"No {step} samples exist for {battery}.")
    if max_curves <= 0 or len(rows) <= max_curves:
        return rows
    if sampling == "random":
        rng = np.random.default_rng(seed)
        index = np.sort(rng.choice(len(rows), size=max_curves, replace=False))
    else:
        index = np.unique(np.round(np.linspace(0, len(rows) - 1, max_curves)).astype(int))
    return rows.iloc[index].reset_index(drop=True)


# =============================================================================
# FIGURE
# =============================================================================

def save_figure(fig, output_stem: Path) -> None:
    import matplotlib.pyplot as plt
    fig.savefig(output_stem.with_suffix(".png"), dpi=PNG_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(output_stem.with_suffix(".svg"), format="svg", bbox_inches="tight", facecolor="white")
    fig.savefig(output_stem.with_suffix(".pdf"), format="pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_life_evolution(signals: np.ndarray, lives: np.ndarray, title: str, output_stem: Path) -> None:
    """signals [N, 2, 512] in V / Ah, lives [N]: three panels coloured by cycle number."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    cmap = plt.get_cmap(COLORMAP)
    norm = Normalize(vmin=float(lives.min()), vmax=float(max(lives.max(), lives.min() + 1)))
    fig, axes = plt.subplots(1, 3, figsize=(19.0, 5.8))
    x = np.arange(signals.shape[-1])
    for signal, life in zip(signals, lives):
        color = cmap(norm(life))
        axes[0].plot(signal[1], signal[0], color=color, linewidth=1.1, alpha=0.9)
        axes[1].plot(x, signal[0], color=color, linewidth=1.1, alpha=0.9)
        axes[2].plot(x, signal[1], color=color, linewidth=1.1, alpha=0.9)
    labels = [("Local Capacity (Ah)", "Voltage (V)"), ("Normalized Time Index", "Voltage (V)"),
              ("Normalized Time Index", "Local Capacity (Ah)")]
    for ax, (xlabel, ylabel) in zip(axes, labels):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        sns.despine(ax=ax)
    fig.suptitle(title)
    fig.tight_layout()
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=list(axes), pad=0.015, fraction=0.025)
    colorbar.set_label("Cycle Number (life index)")
    save_figure(fig, output_stem)


# =============================================================================
# MAIN
# =============================================================================

def print_overview() -> None:
    print("\n" + "=" * 72)
    print("INTERACTIVE BATTERY CURVES-OVER-LIFE VISUALIZER")
    print("=" * 72)
    print("Draws the real charge or discharge segments of one battery from early")
    print("to late life, coloured by cycle number: voltage vs local capacity and")
    print("both channels vs the sample index (V, Ah). Real data only.")
    for dataset in DATASETS:
        status = "ready" if (processed_dir(dataset) / "metadata.csv").exists() else "missing files"
        print(f"  {DATASET_LABELS[dataset]:<9} {processed_dir(dataset).name}/  ({status})")


def list_contents() -> None:
    print_overview()
    for dataset in DATASETS:
        metadata = load_metadata(dataset)
        summary = metadata.groupby("battery_id").agg(first=("life_index", "min"), last=("life_index", "max"),
                                                     n=("file", "size"))
        print(f"\n{DATASET_LABELS[dataset]}: {len(metadata)} segments, {len(summary)} batteries "
              "(life index range, segments)")
        print("  " + ", ".join(f"{b} ({int(r['first'])}-{int(r['last'])}, {int(r['n'])})"
                               for b, r in summary.iterrows()))


def run(dataset: str, battery: str, step: str, max_curves: int = MAX_CURVES, sampling: str = "even",
        seed: int = RANDOM_SEED, output_dir: Path = OUTPUT_DIR):
    configure_plot_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(dataset)
    rows = select_rows(metadata, battery, step, max_curves, sampling, seed)
    signals = np.stack([load_physical_signal(dataset, str(f)) for f in rows["file"]])
    lives = rows["life_index"].to_numpy(np.int64)
    title = (f"{DATASET_LABELS[dataset]} | {battery} | {step.capitalize()} | "
             f"{len(rows)} segments, cycles {lives.min()}-{lives.max()}")
    stem = output_dir / f"{dataset}_{battery}_{step}_life_evolution"
    plot_life_evolution(signals, lives, title, stem)
    rows[["battery_id", "step_type", "life_index", "file"]].to_csv(
        stem.parent / f"{stem.name}_records.csv", index=False, encoding="utf-8-sig")
    print(f"\n{title}")
    print("\nCompleted successfully.")
    print(f"Output directory: {output_dir}")
    print(f"  - {stem.name}.png/.svg/.pdf")
    print(f"  - {stem.name}_records.csv")
    return signals, lives


def main() -> None:
    parser = argparse.ArgumentParser(description="Real segments of one battery over its life, coloured by cycle number.")
    parser.add_argument("--list", action="store_true", help="list datasets, batteries and life ranges, then exit")
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--battery")
    parser.add_argument("--step", choices=["charge", "discharge"])
    parser.add_argument("--max-curves", type=int, default=None,
                        help=f"number of segments drawn (default {MAX_CURVES}; 0 = all)")
    parser.add_argument("--sampling", choices=["even", "random"], default="even",
                        help="how segments are thinned when there are more than --max-curves")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="seed for --sampling random")
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR, help="output directory")
    args = parser.parse_args()
    matplotlib.use("Agg")

    if args.list:
        list_contents()
        return
    print_overview()
    dataset = args.dataset or ask_choice("\nSelect a dataset (calce/nasa/severson): ", list(DATASETS))
    metadata = load_metadata(dataset)
    battery = args.battery or choose_battery(metadata)
    if battery not in set(metadata["battery_id"]):
        raise ValueError(f"Battery {battery} is not in the {DATASET_LABELS[dataset]} metadata.")
    steps = sorted(metadata.loc[metadata["battery_id"] == battery, "step_type"].unique())
    step = args.step or ask_choice(f"\nSelect a step type ({'/'.join(steps)}): ", steps)
    max_curves = args.max_curves
    if max_curves is None:
        max_curves = MAX_CURVES if args.dataset else ask_int(
            f"\nHow many segments should be drawn? (0 = all, Enter = {MAX_CURVES}): ", 0, 100000, MAX_CURVES)
    run(dataset, battery, step, max_curves, args.sampling, args.seed, args.out)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by the user.")
    except Exception as error:
        print("\nThe program could not complete the requested operation.")
        print(f"Reason: {error}")
        print("Please verify the dataset folders and processed files, then run the program again.")
        raise SystemExit(1)
