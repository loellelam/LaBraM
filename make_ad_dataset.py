# Builds LaBraM-compatible train/val/test splits from preprocessed subject H5 files.
# Splits are done at the SUBJECT level — all windows from a subject go entirely
# into one split, preventing data leakage.

import h5py
import json
import numpy as np
import csv
from pathlib import Path
from sklearn.model_selection import train_test_split

# ---- edit these if your paths differ ----
OUTPUT_DIR = Path("output")
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
PARTICIPANTS_TSV = Path("data/participants.tsv")
LABRAM_DATA_DIR = Path("labram_data")
# -----------------------------------------

# Load participant labels from BIDS participants.tsv
# ds004504 uses 'Group' column: 'A' = Alzheimer, 'C' = Control, 'F' = FTD
subject_labels = {}
with open(PARTICIPANTS_TSV) as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        sub = row["participant_id"].replace("sub-", "")
        group = row["Group"].strip().upper()
        if group == "A":
            subject_labels[sub] = 0   # AD
        elif group == "F":
            subject_labels[sub] = 1   # FTD
        elif group == "C":
            subject_labels[sub] = 2   # Control

# Load manifest to find per-subject H5 paths
with open(MANIFEST_PATH) as f:
    manifest = json.load(f)

# Build a dict: subject_id -> (X array, label)
subject_data = {}
all_channels = None

for record in manifest["results"]:
    h5_path = Path(record["output_h5"])
    stem = h5_path.stem  # e.g. "sub-001_task-eyesclosed_labram"

    # extract subject ID
    sub_id = None
    for part in stem.split("_"):
        if part.startswith("sub-"):
            sub_id = part.replace("sub-", "")
            break

    if sub_id is None or sub_id not in subject_labels:
        print(f"Skipping {h5_path.name} — no label found in participants.tsv")
        continue

    with h5py.File(h5_path, "r") as f:
        X = f["X"][:]  # (n_windows, n_channels, n_times)
        channels = [c.decode() for c in f["channels"][:]]

    if all_channels is None:
        all_channels = channels

    label = subject_labels[sub_id]
    subject_data[sub_id] = (X, label)
    label_names = {0: "AD", 1: "FTD", 2: "Control"}
    print(f"  sub-{sub_id}  label={label_names[label]}  windows={X.shape[0]}")

# ---- Subject-level split ----
subjects = list(subject_data.keys())
labels_per_subject = [subject_data[s][1] for s in subjects]

print(f"\nTotal subjects: {len(subjects)}")
print(f"  AD:      {sum(l == 0 for l in labels_per_subject)}")
print(f"  FTD:     {sum(l == 1 for l in labels_per_subject)}")
print(f"  Control: {sum(l == 2 for l in labels_per_subject)}")

# 70% train / 15% val / 15% test, stratified by subject label
subs_trainval, subs_test = train_test_split(
    subjects, test_size=0.15,
    stratify=labels_per_subject,
    random_state=42
)
subs_train, subs_val = train_test_split(
    subs_trainval, test_size=0.15 / 0.85,
    stratify=[subject_data[s][1] for s in subs_trainval],
    random_state=42
)

print(f"\nSubject-level split:")
print(f"  Train: {len(subs_train)} subjects -> {[s for s in subs_train]}")
print(f"  Val:   {len(subs_val)} subjects -> {[s for s in subs_val]}")
print(f"  Test:  {len(subs_test)} subjects -> {[s for s in subs_test]}")

# ---- Concatenate windows per split ----
def build_split(subject_ids):
    X_parts, y_parts = [], []
    for s in subject_ids:
        X, label = subject_data[s]
        X_parts.append(X)
        y_parts.append(np.full(X.shape[0], label, dtype=np.int64))
    return np.concatenate(X_parts), np.concatenate(y_parts)

X_train, y_train = build_split(subs_train)
X_val,   y_val   = build_split(subs_val)
X_test,  y_test  = build_split(subs_test)

# ---- Save ----
LABRAM_DATA_DIR.mkdir(exist_ok=True)

for split_name, X_split, y_split, subs in [
    ("train", X_train, y_train, subs_train),
    ("val",   X_val,   y_val,   subs_val),
    ("test",  X_test,  y_test,  subs_test),
]:
    out = LABRAM_DATA_DIR / f"{split_name}.h5"
    with h5py.File(out, "w") as f:
        f.create_dataset("X", data=X_split.astype(np.float32), compression="gzip")
        f.create_dataset("y", data=y_split)
        # store which subjects are in this split for traceability
        f.attrs["subjects"] = json.dumps(subs)
    ad_count = (y_split == 0).sum()
    ftd_count = (y_split == 1).sum()
    ctrl_count = (y_split == 2).sum()
    print(
        f"\n{split_name}: {len(y_split)} windows | "
        f"AD={ad_count} ({100*ad_count/len(y_split):.1f}%) | "
        f"FTD={ftd_count} ({100*ftd_count/len(y_split):.1f}%) | "
        f"Control={ctrl_count} ({100*ctrl_count/len(y_split):.1f}%)"
    )
with open(LABRAM_DATA_DIR / "channel_names.json", "w") as f:
    json.dump(all_channels, f)

# Save split metadata for reference
split_meta = {
    "split_strategy": "subject_level",
    "train_subjects": subs_train,
    "val_subjects": subs_val,
    "test_subjects": subs_test,
    "label_map": {"0": "AD", "1": "FTD", "2": "Control"}
}
with open(LABRAM_DATA_DIR / "split_info.json", "w") as f:
    json.dump(split_meta, f, indent=2)

print("\nDone. Split info saved to labram_data/split_info.json")