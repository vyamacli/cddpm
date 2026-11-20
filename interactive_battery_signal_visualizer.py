"""
===============================================================================
Interactive CALCE, NASA and Severson Battery Signal Visualizer
===============================================================================

Authors:
    Volkan Yamaçlı, Timur Lale, Serdar Ekinci, Rıdvan Fırat Çınar,
    Davut İzci, Gökhan Yüksek

Article:
    Conditional Diffusion for Probabilistic and Controllable Synthetic
    Battery Charge-Discharge Curves

Purpose:
    Interactively select battery samples from the processed CALCE, NASA and
    Severson datasets and visualize one random real charge segment and one
    random real discharge segment for every selected battery, in physical
    units (voltage in V, local capacity in Ah), one panel per dataset.

Outputs:
    selected_voltage_signals.png / .svg / .pdf
    selected_capacity_signals.png / .svg / .pdf
    selected_signal_records.csv

Usage:
    python interactive_battery_signal_visualizer.py                    (interactive)
    python interactive_battery_signal_visualizer.py --list
    python interactive_battery_signal_visualizer.py --count 2          (first 2 batteries of each dataset)
    python interactive_battery_signal_visualizer.py --select calce=CX2_38 --select severson=b1c0,b2c1
===============================================================================
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable

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

MAX_BATTERIES_PER_DATASET = 10
RANDOM_SEED = 42
PNG_DPI = 300
EXPECTED_CHANNELS = 2
CHANNEL_INDEX = {"voltage": 0, "capacity_local": 1}
CHANNEL_LABELS = {"voltage": "Voltage (V)", "capacity_local": "Local Capacity (Ah)"}
# Battery-level split of CALCE (the CALCE metadata.csv has no split column)
CALCE_SPLIT = {"CX2_33": "train", "CX2_34": "train", "CX2_35": "train", "CX2_36": "train",
               "CX2_37": "val", "CX2_38": "test"}


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
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "0.8",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


# =============================================================================
# USER INPUT
# =============================================================================

def parse_numbered_selection(raw_value: str, number_of_options: int) -> list[int]:
    cleaned = raw_value.strip()
    if not cleaned:
        raise ValueError("No selection was entered.")
    parts = [part.strip() for part in cleaned.split(",")]
    if any(not part for part in parts):
        raise ValueError("The selection contains an empty item. Use a format such as 1,3,5.")
    try:
        selected_numbers = [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError("All selections must be whole numbers separated by commas.") from exc
    if len(set(selected_numbers)) != len(selected_numbers):
        raise ValueError("Duplicate selections are not allowed.")
    invalid = [n for n in selected_numbers if n < 1 or n > number_of_options]
    if invalid:
        raise ValueError("The following selection numbers are outside the available range: "
                         + ", ".join(map(str, invalid)))
    return [n - 1 for n in selected_numbers]


def request_datasets() -> list[str]:
    while True:
        raw = input("\nWhich datasets should be shown? (calce,nasa,severson or all): ").strip().lower()
        chosen = list(DATASETS) if raw in ("", "all") else [p.strip() for p in raw.split(",") if p.strip()]
        if chosen and all(c in DATASETS for c in chosen) and len(set(chosen)) == len(chosen):
            return chosen
        print("Invalid input. Use dataset names separated by commas, for example calce,severson.")


def request_battery_count(maximum_available: int) -> int:
    maximum_allowed = min(MAX_BATTERIES_PER_DATASET, maximum_available)
    while True:
        raw_value = input(f"\nHow many battery samples would you like to select from each "
                          f"dataset? (1-{maximum_allowed}): ").strip()
        try:
            count = int(raw_value)
        except ValueError:
            print("Invalid input. Please enter a whole number, for example 3.")
            continue
        if 1 <= count <= maximum_allowed:
            return count
        print(f"Please enter a value between 1 and {maximum_allowed}.")


def request_battery_selection(dataset_label: str, battery_ids: list[str], metadata: pd.DataFrame,
                              required_count: int) -> list[str]:
    print("\n" + "=" * 72)
    print(f"{dataset_label} BATTERIES")
    print("=" * 72)
    splits = metadata.groupby("battery_id")["split"].first()
    for index, battery_id in enumerate(battery_ids, start=1):
        print(f"{index:>3}. {battery_id:<8} ({splits.get(battery_id, 'unknown')})")
    while True:
        raw_value = input(f"\nSelect exactly {required_count} {dataset_label} battery "
                          "number(s), separated by commas: ")
        try:
            selected_indices = parse_numbered_selection(raw_value, len(battery_ids))
        except ValueError as error:
            print(f"Invalid selection: {error}")
            print("Please try again.")
            continue
        if len(selected_indices) != required_count:
            print(f"You must select exactly {required_count} battery sample(s), "
                  f"but {len(selected_indices)} were entered.")
            print("Please try again.")
            continue
        return [battery_ids[index] for index in selected_indices]


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
    if "split" not in metadata.columns:
        metadata["split"] = metadata["battery_id"].map(CALCE_SPLIT).fillna("unknown")
    if "test_set" in metadata.columns:
        is_test = metadata["split"] == "test"
        metadata.loc[is_test, "split"] = "test_" + metadata.loc[is_test, "test_set"].astype(str)
    metadata = metadata[metadata["step_type"].isin(["charge", "discharge"])].copy()
    if metadata.empty:
        raise ValueError(f"No valid charge/discharge rows were found in {DATASET_LABELS[dataset]} metadata.")
    return metadata


def load_physical_signal(dataset: str, file_name: str) -> np.ndarray:
    sample_path = processed_dir(dataset) / "samples" / file_name
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample file was not found: {sample_path}")
    with np.load(sample_path) as data:
        if "x_raw" not in data:
            raise KeyError(f"'x_raw' was not found in: {sample_path}")
        signal = data["x_raw"].astype(np.float32)
    if signal.ndim != 2 or signal.shape[0] != EXPECTED_CHANNELS:
        raise ValueError(f"Unexpected x_raw shape in {sample_path}: {signal.shape}")
    if not np.all(np.isfinite(signal)):
        raise ValueError(f"NaN or Inf values were found in {sample_path}")
    return signal


# =============================================================================
# RANDOM SAMPLE SELECTION
# =============================================================================

def select_random_cycle(metadata: pd.DataFrame, battery_id: str, step_type: str,
                        rng: random.Random) -> pd.Series:
    candidates = metadata[(metadata["battery_id"] == battery_id) & (metadata["step_type"] == step_type)]
    if candidates.empty:
        raise ValueError(f"No {step_type} sample is available for battery {battery_id}.")
    return candidates.iloc[rng.randrange(len(candidates))]


def collect_selected_records(dataset: str, metadata: pd.DataFrame, selected_batteries: Iterable[str],
                             rng: random.Random) -> list[dict]:
    records: list[dict] = []
    for battery_id in selected_batteries:
        for step_type in ("charge", "discharge"):
            row = select_random_cycle(metadata, battery_id, step_type, rng)
            records.append({
                "dataset": DATASET_LABELS[dataset],
                "battery_id": battery_id,
                "step_type": step_type,
                "life_index": int(row["life_index"]),
                "split": str(row.get("split", "unknown")),
                "file": str(row["file"]),
                "signal": load_physical_signal(dataset, str(row["file"])),
            })
    return records


# =============================================================================
# FIGURES
# =============================================================================

def save_figure(fig, output_stem: Path) -> None:
    import matplotlib.pyplot as plt
    fig.savefig(output_stem.with_suffix(".png"), dpi=PNG_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(output_stem.with_suffix(".svg"), format="svg", bbox_inches="tight", facecolor="white")
    fig.savefig(output_stem.with_suffix(".pdf"), format="pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_channel(records: list[dict], channel_name: str, output_stem: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    channel_index = CHANNEL_INDEX[channel_name]
    datasets = list(dict.fromkeys(r["dataset"] for r in records))
    fig, axes = plt.subplots(1, len(datasets), figsize=(7.0 * len(datasets), 6.0), squeeze=False)
    line_style_map = {"charge": "-", "discharge": "--"}
    for ax, dataset in zip(axes[0], datasets):
        subset = [r for r in records if r["dataset"] == dataset]
        batteries = list(dict.fromkeys(r["battery_id"] for r in subset))
        colors = dict(zip(batteries, sns.color_palette("husl", len(batteries))))
        for record in subset:
            signal = record["signal"][channel_index]
            ax.plot(np.arange(len(signal)), signal, color=colors[record["battery_id"]],
                    linestyle=line_style_map[record["step_type"]], linewidth=2.0, alpha=0.95,
                    label=f"{record['battery_id']} | {record['step_type'].capitalize()} | "
                          f"LifeCycle {record['life_index']}")
        ax.set_title(dataset)
        ax.set_xlabel("Normalized Time Index")
        ax.set_ylabel(CHANNEL_LABELS[channel_name])
        ax.legend(title="Battery | Step | LifeCycle Index", fontsize=8, title_fontsize=9, loc="best")
        sns.despine(ax=ax)
    fig.tight_layout()
    save_figure(fig, output_stem)


def save_selected_record_table(records: list[dict], output_path: Path) -> None:
    rows = [{k: r[k] for k in ("dataset", "battery_id", "step_type", "life_index", "split", "file")}
            for r in records]
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")


# =============================================================================
# MAIN
# =============================================================================

def print_overview() -> None:
    print("\n" + "=" * 72)
    print("INTERACTIVE CALCE, NASA AND SEVERSON BATTERY SIGNAL VISUALIZER")
    print("=" * 72)
    print("Plots one random real charge and one random real discharge segment")
    print("(512 points, voltage in V and local capacity in Ah) for every selected")
    print("battery of the selected datasets.")
    for dataset in DATASETS:
        status = "ready" if (processed_dir(dataset) / "metadata.csv").exists() else "missing files"
        print(f"  {DATASET_LABELS[dataset]:<9} {processed_dir(dataset).name}/  ({status})")


def list_contents() -> None:
    print_overview()
    for dataset in DATASETS:
        metadata = load_metadata(dataset)
        per_battery = metadata.groupby("battery_id").agg(split=("split", "first"), n=("file", "size"))
        print(f"\n{DATASET_LABELS[dataset]}: {len(metadata)} segments, {len(per_battery)} batteries")
        for split, group in per_battery.groupby("split"):
            print(f"  {split:<15} " + ", ".join(group.index))


def parse_select(values: list[str]) -> dict[str, list[str]]:
    selection: dict[str, list[str]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--select expects dataset=BATTERY[,BATTERY...], got {value!r}")
        dataset, batteries = value.split("=", 1)
        dataset = dataset.strip().lower()
        if dataset not in DATASETS:
            raise ValueError(f"Unknown dataset in --select: {dataset}")
        selection[dataset] = [b.strip() for b in batteries.split(",") if b.strip()]
    return selection


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot real charge/discharge segments of selected batteries.")
    parser.add_argument("--list", action="store_true", help="list datasets and batteries per split, then exit")
    parser.add_argument("--datasets", help="comma-separated subset of calce,nasa,severson")
    parser.add_argument("--count", type=int, help="take the first N batteries (sorted) of each dataset")
    parser.add_argument("--select", action="append", default=[], help="dataset=BATTERY[,BATTERY...] (repeatable)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR, help="output directory")
    args = parser.parse_args()
    matplotlib.use("Agg")

    if args.list:
        list_contents()
        return
    print_overview()
    configure_plot_style()
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    selection = parse_select(args.select)
    if selection:
        datasets = list(selection)
    elif args.datasets:
        datasets = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]
    elif args.count:
        datasets = list(DATASETS)
    else:
        datasets = request_datasets()
    if any(d not in DATASETS for d in datasets):
        raise ValueError(f"Datasets must be among {', '.join(DATASETS)}.")

    metadata = {d: load_metadata(d) for d in datasets}
    battery_ids = {d: sorted(metadata[d]["battery_id"].unique()) for d in datasets}
    if not selection:
        count = args.count or request_battery_count(min(len(v) for v in battery_ids.values()))
        for d in datasets:
            selection[d] = (battery_ids[d][:count] if args.count else
                            request_battery_selection(DATASET_LABELS[d], battery_ids[d], metadata[d], count))
    for d, batteries in selection.items():
        unknown = [b for b in batteries if b not in battery_ids[d]]
        if unknown:
            raise ValueError(f"Unknown {DATASET_LABELS[d]} batteries: {', '.join(unknown)}")

    print("\nSelecting random charge and discharge cycles...")
    records: list[dict] = []
    for d in datasets:
        records.extend(collect_selected_records(d, metadata[d], selection[d], rng))

    plot_channel(records, "voltage", output_dir / "selected_voltage_signals")
    plot_channel(records, "capacity_local", output_dir / "selected_capacity_signals")
    save_selected_record_table(records, output_dir / "selected_signal_records.csv")

    print("\nCompleted successfully.")
    print(f"Output directory: {output_dir}")
    print("Generated figures:")
    print("  - selected_voltage_signals.png/.svg/.pdf")
    print("  - selected_capacity_signals.png/.svg/.pdf")
    print("Selected sample metadata:")
    print("  - selected_signal_records.csv")


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
