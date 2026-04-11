# make_ad_dataset.py
# Run this AFTER your preprocessing script has populated output/
# It reads all subject H5s, attaches AD/control labels from ds004504 
# participant metadata, and writes train/val/test splits.

import h5py
import json
import numpy as np
from pathlib import Path

# ---- edit these ----
OUTPUT_DIR = Path("output")           # where your preprocessor wrote .h5 files
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
PARTICIPANTS_TSV = Path("data/participants.tsv")  # ds004504 BIDS participants file
LABRAM_DATA_DIR = Path("labram_data")
# --------------------

# ds004504 uses 'Group' column: 'A' = Alzheimer, 'C' = Control
# Load participant labels
import csv
subject_labels = {}
with open(PARTICIPANTS_TSV) as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        sub = row["participant_id"].replace("sub-", "")
        label = 1 if row["Group"].strip().upper() == "A" else 0
        subject_labels[sub] = label

# Load manifest to find per-subject H5 paths
with open(MANIFEST_PATH) as f:
    manifest = json.load(f)

all_X, all_y, all_channels = [], [], None

for record in manifest["results"]:
    h5_path = Path(record["output_h5"])
    # infer subject ID from filename e.g. "sub-001_labram.h5"
    stem = h5_path.stem  # "sub-001_labram"
    sub_id = None
    for part in stem.split("_"):
        if part.startswith("sub-"):
            sub_id = part.replace("sub-", "")
            break
    if sub_id is None or sub_id not in subject_labels:
        print(f"Skipping {h5_path.name} — no label found")
        continue

    label = subject_labels[sub_id]
    with h5py.File(h5_path, "r") as f:
        X = f["X"][:]               # (n_windows, n_channels, n_times)
        channels = [c.decode() for c in f["channels"][:]]

    y = np.full(X.shape[0], label, dtype=np.int64)
    all_X.append(X)
    all_y.append(y)
    if all_channels is None:
        all_channels = channels
    print(f"  sub-{sub_id} label={label}  windows={X.shape[0]}")

X_all = np.concatenate(all_X, axis=0)   # (total_windows, n_ch, n_times)
y_all = np.concatenate(all_y, axis=0)   # (total_windows,)
print(f"\nTotal windows: {X_all.shape[0]}  channels: {len(all_channels)}")

# Train / val / test split (70 / 15 / 15), stratified by label
from sklearn.model_selection import train_test_split
idx = np.arange(len(y_all))
idx_tv, idx_test = train_test_split(idx, test_size=0.15, stratify=y_all, random_state=42)
idx_train, idx_val = train_test_split(idx_tv, test_size=0.15/0.85, stratify=y_all[idx_tv], random_state=42)

LABRAM_DATA_DIR.mkdir(exist_ok=True)
for split_name, split_idx in [("train", idx_train), ("val", idx_val), ("test", idx_test)]:
    out = LABRAM_DATA_DIR / f"{split_name}.h5"
    with h5py.File(out, "w") as f:
        f.create_dataset("X", data=X_all[split_idx].astype(np.float32), compression="gzip")
        f.create_dataset("y", data=y_all[split_idx])
    print(f"Wrote {out}  shape={X_all[split_idx].shape}  pos={y_all[split_idx].sum()}")

# Save channel list — LaBraM needs this separately
with open(LABRAM_DATA_DIR / "channel_names.json", "w") as f:
    json.dump(all_channels, f)
print("\nDone. Channel names saved to labram_data/channel_names.json")