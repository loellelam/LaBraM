"""
Universal EEG -> LaBraM preprocessing script.

What it does
------------
- Accepts either:
    1) a single raw EEG file
    2) a folder containing raw EEG files
    3) a BIDS root (optional)
- Auto-detects common EEG file formats supported by MNE
- Keeps EEG channels, optionally picks a fixed ordered subset
- Filters / notches / resamples for LaBraM-style preprocessing
- Converts to microvolts (uV)
- Cuts into fixed-length windows
- Optionally rejects very noisy windows by peak-to-peak amplitude
- Saves one HDF5 file per recording plus a manifest JSON

This is meant to be a practical bridge from arbitrary raw EEG files
into a consistent windowed format that is easier to feed into a
custom LaBraM dataset loader.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Iterable

import h5py
import mne
import numpy as np

# Optional WFDB support for PhysioNet-style .hea/.mat or .hea/.dat records
try:
    import wfdb
    HAVE_WFDB = True
except Exception:
    HAVE_WFDB = False

# Optional BIDS support
try:
    from mne_bids import BIDSPath, get_entity_vals, read_raw_bids
    HAVE_MNE_BIDS = True
except Exception:
    HAVE_MNE_BIDS = False

# -----------------------------
# CONFIG: EDIT THESE FIRST
# -----------------------------
# Set INPUT_PATH to one of:
# - a single EEG file
# - a directory containing EEG files
# - a BIDS root directory (set INPUT_MODE = "bids")
INPUT_PATH = "data/"
INPUT_MODE = "auto"   # "auto", "files", or "bids"
OUT_DIR = "output"

# For BIDS mode only
BIDS_TASK = None       # e.g. "eyesclosed". Use None to process all tasks found.

# LaBraM-style preprocessing
RESAMPLE_HZ = 200
FMIN = 0.1
FMAX = 75.0
NOTCH_HZ = 50.0        # change to 50.0 if the recordings were made in a 50 Hz mains region
APPLY_AVERAGE_REFERENCE = False

# Windowing
WINDOW_SEC = 4.0
WINDOW_OVERLAP_SEC = 2.0

# Basic artifact rejection on each window, based on peak-to-peak amplitude in microvolts.
# Set to None to disable.
REJECT_PEAK_TO_PEAK_UV = 250.0

# Optional fixed channel subset / order.
# Leave as None to keep all EEG channels in the order provided by the file.
CHANNELS_TO_KEEP = None
# Example:
# CHANNELS_TO_KEEP = [
#     "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8",
#     "T3", "C3", "CZ", "C4", "T4",
#     "T5", "P3", "PZ", "P4", "T6", "O1", "O2"
# ]

# Common EEG extensions MNE can read with dedicated readers.

# Optional hints for detecting EEG channels in WFDB records.
# Any channel whose name contains one of these (case-insensitive) will be marked as EEG.
WFDB_EEG_NAME_HINTS = ["EEG", "FP", "FZ", "CZ", "PZ", "O1", "O2", "F3", "F4", "C3", "C4", "P3", "P4", "T3", "T4", "T5", "T6"]

# If a WFDB record has no channels matching the hints above, treat all channels as EEG.
WFDB_ASSUME_ALL_EEG_IF_NO_HINTS = True

SUPPORTED_EXTENSIONS = {
    ".hea",
    ".set",
    ".edf",
    ".bdf",
    ".gdf",
    ".cnt",
    ".vhdr",
    ".egi",
    ".mff",
    ".data",
    ".nxe",
    ".lay",
    ".eeg",    # used by Nihon Kohden and BrainVision companion files; handled carefully below
    ".fif",
    ".fif.gz",
}


def normalize_channel_name(ch: str) -> str:
    """Normalize channel labels to something stable for downstream use."""
    ch = str(ch).strip().upper()
    for prefix in ("EEG ",):
        if ch.startswith(prefix):
            ch = ch[len(prefix):]
    for suffix in ("-REF", "-LE"):
        if ch.endswith(suffix):
            ch = ch[: -len(suffix)]
    return ch


# -----------------------------
# File readers
# -----------------------------
def _safe_reader(fn: Callable, *args, **kwargs):
    return fn(*args, **kwargs)


def read_wfdb_as_raw(path: str | Path) -> mne.io.BaseRaw:
    """Read a WFDB record (.hea with companion .mat/.dat) and convert it to MNE RawArray."""
    if not HAVE_WFDB:
        raise ImportError(
            "wfdb is required to read .hea WFDB records. Install it with: pip install wfdb"
        )

    path = Path(path)
    if path.suffix.lower() != ".hea":
        raise ValueError(f"WFDB reader expects a .hea header file, got: {path}")

    record = wfdb.rdrecord(str(path.with_suffix('')))
    data = getattr(record, 'p_signal', None)
    if data is None:
        data = getattr(record, 'd_signal', None)
    if data is None:
        raise RuntimeError(f"WFDB record did not contain readable signal data: {path}")

    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise RuntimeError(f"Expected 2D WFDB signal array, got shape {data.shape} for {path}")

    ch_names = [str(ch) for ch in (record.sig_name or [f"CH{i+1}" for i in range(data.shape[1])])]
    hint_hits = []
    for ch in ch_names:
        up = ch.upper()
        hint_hits.append(any(h in up for h in WFDB_EEG_NAME_HINTS))

    if any(hint_hits):
        ch_types = ["eeg" if hit else "misc" for hit in hint_hits]
    elif WFDB_ASSUME_ALL_EEG_IF_NO_HINTS:
        ch_types = ["eeg"] * len(ch_names)
    else:
        ch_types = ["misc"] * len(ch_names)

    info = mne.create_info(ch_names=ch_names, sfreq=float(record.fs), ch_types=ch_types)
    raw = mne.io.RawArray(data.T, info, verbose=False)
    raw.info["description"] = f"WFDB record loaded from {path.name}"
    return raw


def read_raw_any(path: str | Path) -> mne.io.BaseRaw:
    """Read one raw EEG recording using the best available reader for its extension."""
    path = Path(path)
    suffixes = ''.join(path.suffixes).lower()
    ext = path.suffix.lower()

    # Multi-file formats: use the header/entry file, not the payload file.
    if ext == ".hea":
        raw = read_wfdb_as_raw(path)
    elif ext == ".vhdr":
        raw = mne.io.read_raw_brainvision(path, preload=True, verbose=False)
    elif ext == ".set":
        raw = mne.io.read_raw_eeglab(path, preload=True, verbose=False)
    elif ext == ".edf":
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    elif ext == ".bdf":
        raw = mne.io.read_raw_bdf(path, preload=True, verbose=False)
    elif ext == ".gdf":
        raw = mne.io.read_raw_gdf(path, preload=True, verbose=False)
    elif ext == ".cnt":
        raw = mne.io.read_raw_cnt(path, preload=True, verbose=False)
    elif ext == ".egi" or ext == ".mff":
        raw = mne.io.read_raw_egi(path, preload=True, verbose=False)
    elif ext == ".data":
        raw = mne.io.read_raw_nicolet(path, ch_type="eeg", preload=True, verbose=False)
    elif ext == ".nxe":
        raw = mne.io.read_raw_eximia(path, preload=True, verbose=False)
    elif ext == ".lay":
        raw = mne.io.read_raw_persyst(path, preload=True, verbose=False)
    elif suffixes.endswith(".fif.gz") or ext == ".fif":
        raw = mne.io.read_raw_fif(path, preload=True, verbose=False)
    elif ext == ".eeg":
        # BrainVision .eeg is not the entry point; .vhdr is.
        # Nihon Kohden uses .EEG as the main file.
        vhdr_neighbor = path.with_suffix('.vhdr')
        if vhdr_neighbor.exists():
            raise ValueError(
                f"{path.name} looks like a BrainVision payload file. Use the .vhdr file instead: {vhdr_neighbor}"
            )
        raw = mne.io.read_raw_nihon(path, preload=True, verbose=False)
    else:
        raise ValueError(
            f"Unsupported file type for {path.name}.\n"
            "Try using a supported EEG file such as .set, .edf, .bdf, .gdf, .cnt, .vhdr, .fif, .lay, .mff, .egi, .data, .nxe"
        )
    return raw


# -----------------------------
# Discovery
# -----------------------------
def list_candidate_files(root: str | Path) -> list[Path]:
    root = Path(root)
    if root.is_file():
        return [root]

    candidates: list[Path] = []
    for p in root.rglob('*'):
        if not p.is_file():
            continue
        full_suffix = ''.join(p.suffixes).lower()
        ext = p.suffix.lower()

        # Prefer BrainVision header (.vhdr), skip payload .eeg if .vhdr exists.
        if ext == '.eeg' and p.with_suffix('.vhdr').exists():
            continue
        # WFDB records are entered via .hea, not the companion payload file.
        if ext in {'.mat', '.dat'} and p.with_suffix('.hea').exists():
            continue

        if full_suffix.endswith('.fif.gz') or ext in SUPPORTED_EXTENSIONS:
            candidates.append(p)

    # De-duplicate and sort for stable runs.
    uniq = sorted({p.resolve() for p in candidates})
    return [Path(p) for p in uniq]


# -----------------------------
# Preprocessing
# -----------------------------
def pick_and_order_eeg_channels(raw: mne.io.BaseRaw) -> tuple[mne.io.BaseRaw, list[str]]:
    """Keep EEG channels and optionally enforce a fixed channel subset/order."""
    raw = raw.copy()
    raw.pick(picks="eeg", exclude=[])
    if len(raw.ch_names) == 0:
        raise RuntimeError("No EEG channels found after picking EEG channels.")

    current_normalized = [normalize_channel_name(ch) for ch in raw.ch_names]
    rename_map = {old: new for old, new in zip(raw.ch_names, current_normalized) if old != new}
    if rename_map:
        raw.rename_channels(rename_map)

    if CHANNELS_TO_KEEP is not None:
        desired = [normalize_channel_name(ch) for ch in CHANNELS_TO_KEEP]
        available = set(raw.ch_names)
        keep = [ch for ch in desired if ch in available]
        if not keep:
            raise RuntimeError(
                "None of the requested CHANNELS_TO_KEEP were found in this file. "
                f"Available channels: {raw.ch_names}"
            )
        raw.pick(keep)
        raw.reorder_channels(keep)
        final_order = keep
    else:
        final_order = list(raw.ch_names)

    return raw, final_order


def preprocess_raw(raw: mne.io.BaseRaw) -> tuple[np.ndarray, list[str], int, np.ndarray, dict]:
    """Preprocess one raw recording and return windowed data + metadata."""
    raw = raw.copy()
    raw.load_data()

    # Keep and normalize EEG channels first.
    raw, final_order = pick_and_order_eeg_channels(raw)

    # Filter and line-noise removal.
    raw.filter(FMIN, FMAX, verbose=False)
    if NOTCH_HZ is not None:
        raw.notch_filter(freqs=NOTCH_HZ, verbose=False)

    # Referencing can be dataset-specific; keep optional.
    if APPLY_AVERAGE_REFERENCE:
        raw.set_eeg_reference("average", verbose=False)

    # Resample to LaBraM-friendly 200 Hz.
    raw.resample(RESAMPLE_HZ, verbose=False)

    # Convert from volts to microvolts for export.
    raw._data = raw._data * 1e6

    epochs = mne.make_fixed_length_epochs(
        raw,
        duration=WINDOW_SEC,
        overlap=WINDOW_OVERLAP_SEC,
        preload=True,
        reject_by_annotation=True,
        verbose=False,
    )

    X = epochs.get_data().copy().astype(np.float32)  # (n_windows, n_channels, n_times)
    if X.shape[0] == 0:
        raise RuntimeError("No windows were created. Recording may be too short or fully rejected.")

    starts = epochs.events[:, 0] / epochs.info["sfreq"]
    keep_mask = np.ones(X.shape[0], dtype=bool)
    reject_stats = {
        "initial_windows": int(X.shape[0]),
        "kept_windows": int(X.shape[0]),
        "dropped_windows": 0,
        "reject_peak_to_peak_uv": REJECT_PEAK_TO_PEAK_UV,
    }

    if REJECT_PEAK_TO_PEAK_UV is not None:
        p2p = X.max(axis=-1) - X.min(axis=-1)    # (n_windows, n_channels)
        too_noisy = (p2p > REJECT_PEAK_TO_PEAK_UV).any(axis=1)
        keep_mask = ~too_noisy
        X = X[keep_mask]
        starts = starts[keep_mask]
        reject_stats["kept_windows"] = int(X.shape[0])
        reject_stats["dropped_windows"] = int((~keep_mask).sum())

    if X.shape[0] == 0:
        raise RuntimeError(
            "All windows were rejected as too noisy. Increase REJECT_PEAK_TO_PEAK_UV or disable it by setting None."
        )

    meta = {
        "n_channels": int(X.shape[1]),
        "n_times": int(X.shape[2]),
        "window_sec": float(WINDOW_SEC),
        "window_overlap_sec": float(WINDOW_OVERLAP_SEC),
        "preprocessing": {
            "filter_hz": [float(FMIN), float(FMAX)],
            "notch_hz": None if NOTCH_HZ is None else float(NOTCH_HZ),
            "resample_hz": int(RESAMPLE_HZ),
            "units": "uV",
            "average_reference": bool(APPLY_AVERAGE_REFERENCE),
        },
        "rejection": reject_stats,
    }
    return X, final_order, int(epochs.info["sfreq"]), starts.astype(np.float32), meta


# -----------------------------
# Saving
# -----------------------------
def save_h5(out_path: str | Path, X: np.ndarray, channels: list[str], sfreq: int,
            window_starts: np.ndarray, source_path: str, meta: dict) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("X", data=X, compression="gzip")
        f.create_dataset("channels", data=np.array(channels, dtype="S"))
        f.create_dataset("sfreq", data=sfreq)
        f.create_dataset("window_start_sec", data=window_starts)
        f.create_dataset("source_path", data=str(source_path).encode("utf-8"))
        f.create_dataset("window_sec", data=float(WINDOW_SEC))
        f.create_dataset("window_overlap_sec", data=float(WINDOW_OVERLAP_SEC))
        f.attrs["units"] = "uV"
        f.attrs["format"] = "windowed_eeg_for_labram"
        f.attrs["n_windows"] = int(X.shape[0])
        for k, v in meta["preprocessing"].items():
            f.attrs[f"preproc_{k}"] = "None" if v is None else v


def safe_stem(path: Path) -> str:
    name = path.name
    if name.endswith('.fif.gz'):
        return name[:-7]
    return path.stem


# -----------------------------
# Runners
# -----------------------------
def process_single_file(path: Path) -> dict:
    print(f"Processing file: {path}")
    raw = read_raw_any(path)
    X, channels, sfreq, starts, meta = preprocess_raw(raw)
    out_name = f"{safe_stem(path)}_labram.h5"
    out_path = Path(OUT_DIR) / out_name
    save_h5(out_path, X, channels, sfreq, starts, str(path), meta)
    return {
        "source_path": str(path),
        "output_h5": str(out_path),
        "n_windows": int(X.shape[0]),
        "n_channels": int(X.shape[1]),
        "n_times": int(X.shape[2]),
        "channels": channels,
        **meta,
    }


def process_files_mode(input_path: str | Path) -> tuple[list[dict], list[dict]]:
    files = list_candidate_files(input_path)
    if not files:
        raise RuntimeError(f"No supported EEG files found under: {input_path}")

    results: list[dict] = []
    failures: list[dict] = []
    for p in files:
        try:
            results.append(process_single_file(p))
        except Exception as e:
            failures.append({"source_path": str(p), "error": str(e)})
            print(f"  FAILED: {p}\n    {e}")
    return results, failures


def process_bids_mode(bids_root: str | Path) -> tuple[list[dict], list[dict]]:
    if not HAVE_MNE_BIDS:
        raise RuntimeError("mne-bids is not installed, so BIDS mode is unavailable.")

    bids_root = Path(bids_root)
    subjects = get_entity_vals(bids_root, "subject")
    tasks = get_entity_vals(bids_root, "task")
    if not subjects:
        raise RuntimeError(f"No BIDS subjects found under: {bids_root}")

    task_list = [BIDS_TASK] if BIDS_TASK else (tasks if tasks else [None])

    results: list[dict] = []
    failures: list[dict] = []
    for sub in subjects:
        for task in task_list:
            try:
                bids_path = BIDSPath(subject=sub, task=task, root=bids_root)
                raw = read_raw_bids(bids_path=bids_path, verbose=False)
                X, channels, sfreq, starts, meta = preprocess_raw(raw)
                task_tag = f"_task-{task}" if task else ""
                out_name = f"sub-{sub}{task_tag}_labram.h5"
                out_path = Path(OUT_DIR) / out_name
                save_h5(out_path, X, channels, sfreq, starts, str(bids_path), meta)
                results.append({
                    "subject": sub,
                    "task": task,
                    "source_path": str(bids_path),
                    "output_h5": str(out_path),
                    "n_windows": int(X.shape[0]),
                    "n_channels": int(X.shape[1]),
                    "n_times": int(X.shape[2]),
                    "channels": channels,
                    **meta,
                })
                print(f"Processed BIDS subject={sub}, task={task}")
            except Exception as e:
                failures.append({"subject": sub, "task": task, "error": str(e)})
                print(f"  FAILED BIDS subject={sub}, task={task}: {e}")
    return results, failures


def detect_mode(input_path: str | Path) -> str:
    if INPUT_MODE in {"files", "bids"}:
        return INPUT_MODE
    p = Path(input_path)
    if p.is_file():
        return "files"
    if HAVE_MNE_BIDS and p.is_dir() and (p / "dataset_description.json").exists():
        try:
            subs = get_entity_vals(p, "subject")
            if subs:
                return "bids"
        except Exception:
            pass
    return "files"


def main() -> None:
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    mode = detect_mode(INPUT_PATH)
    print(f"INPUT_MODE resolved to: {mode}")

    if mode == "bids":
        results, failures = process_bids_mode(INPUT_PATH)
    else:
        results, failures = process_files_mode(INPUT_PATH)

    manifest = {
        "input_path": str(INPUT_PATH),
        "mode": mode,
        "out_dir": str(OUT_DIR),
        "config": {
            "resample_hz": RESAMPLE_HZ,
            "filter_hz": [FMIN, FMAX],
            "notch_hz": NOTCH_HZ,
            "window_sec": WINDOW_SEC,
            "window_overlap_sec": WINDOW_OVERLAP_SEC,
            "reject_peak_to_peak_uv": REJECT_PEAK_TO_PEAK_UV,
            "average_reference": APPLY_AVERAGE_REFERENCE,
            "channels_to_keep": CHANNELS_TO_KEEP,
        },
        "results": results,
        "failures": failures,
    }

    manifest_path = Path(OUT_DIR) / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nDone.")
    print(f"Saved manifest: {manifest_path}")
    print(f"Successful recordings: {len(results)}")
    print(f"Failures: {len(failures)}")


if __name__ == "__main__":
    main()
