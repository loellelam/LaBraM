# Builds LaBraM-compatible H5 dataset from CAUEEG annotations.json
# Binary classification:
#   1 = AD
#   0 = non-AD
#
# Output:
#   labram_data/test.h5
#   labram_data/channel_names.json
#   labram_data/dataset_info.json

import os, shutil
import h5py
import json
import numpy as np
from pathlib import Path

# ---------------- PATHS ----------------
OUTPUT_DIR = Path("output")
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"

ANNOTATIONS_JSON = Path("data/caueeg-dataset/annotation.json")

LABRAM_DATA_DIR = Path("labram_data")
# ---------------------------------------


# ============================================================
# LOAD LABELS FROM annotations.json
# ============================================================

subject_labels = {}

with open(ANNOTATIONS_JSON, "r") as f:
    annotations = json.load(f)

# annotations["data"] is a list of subjects
for item in annotations["data"]:

    serial = item["serial"]          # e.g. "00004"
    symptoms = item.get("symptom", [])

    # lowercase everything for safety
    symptoms = [s.lower() for s in symptoms]

    # -----------------------------
    # Binary labels:
    # AD = 1
    # non-AD = 0
    # -----------------------------
    if "ad" in symptoms:
        label = 1
    else:
        label = 0

    subject_labels[serial] = label

print(f"Loaded labels for {len(subject_labels)} subjects")


# ============================================================
# LOAD MANIFEST
# ============================================================

with open(MANIFEST_PATH, "r") as f:
    manifest = json.load(f)


# ============================================================
# BUILD DATASET
# ============================================================

X_parts = []
y_parts = []

all_channels = None
included_subjects = []

for record in manifest["results"]:

    h5_path = Path(record["output_h5"])

    # Example filename:
    # sub-00001_task-eyesclosed_labram.h5
    stem = h5_path.stem

    # --------------------------------------------------------
    # EXTRACT SUBJECT ID
    # --------------------------------------------------------
    sub_id = stem.removesuffix("_labram")

    if not sub_id:
        print(f"Skipping {h5_path.name} — no subject ID found")
        continue

    if sub_id not in subject_labels:
        print(f"Skipping {h5_path.name} — no label in annotations.json")
        continue

    label = subject_labels[sub_id]

    # --------------------------------------------------------
    # LOAD EEG WINDOWS
    # --------------------------------------------------------
    with h5py.File(h5_path, "r") as f:

        # Shape:
        # (n_windows, n_channels, n_times)
        X = f["X"][:]

        channels = [c.decode() for c in f["channels"][:]]

    # Save channel names once
    if all_channels is None:
        all_channels = channels

    # Create labels per window
    y = np.full(X.shape[0], label, dtype=np.int64)

    X_parts.append(X)
    y_parts.append(y)

    included_subjects.append(sub_id)

    label_name = "AD" if label == 1 else "non-AD"

    print(
        f"sub-{sub_id} | "
        f"label={label_name} | "
        f"windows={X.shape[0]}"
    )


# ============================================================
# CONCATENATE ALL SUBJECTS
# ============================================================

X_all = np.concatenate(X_parts)
y_all = np.concatenate(y_parts)

print("\nFinal dataset:")
print(f"  Total windows: {len(y_all)}")
print(f"  non-AD: {(y_all == 0).sum()}")
print(f"  AD:     {(y_all == 1).sum()}")


# ============================================================
# SAVE DATASET
# ============================================================

print("\nSaving...")

# Make empty LABRAM_DATA_DIR directory
shutil.rmtree(LABRAM_DATA_DIR)
os.makedirs(LABRAM_DATA_DIR)

out_path = LABRAM_DATA_DIR / "test.h5"

with h5py.File(out_path, "w") as f:

    f.create_dataset(
        "X",
        data=X_all.astype(np.float32),
        compression="gzip"
    )

    f.create_dataset(
        "y",
        data=y_all
    )

    # metadata
    f.attrs["subjects"] = json.dumps(included_subjects)

print(f"\nSaved dataset to: {out_path}")


# ============================================================
# SAVE CHANNEL NAMES
# ============================================================

with open(LABRAM_DATA_DIR / "channel_names.json", "w") as f:
    json.dump(all_channels, f, indent=2)

print("Saved channel_names.json")


# ============================================================
# SAVE DATASET METADATA
# ============================================================

dataset_info = {
    "dataset": "CAUEEG",
    "classification": "binary",
    "label_map": {
        "0": "non-AD",
        "1": "AD"
    },
    "n_subjects": len(included_subjects),
    "n_windows": int(len(y_all)),
    "subjects": included_subjects
}

with open(LABRAM_DATA_DIR / "dataset_info.json", "w") as f:
    json.dump(dataset_info, f, indent=2)

print("Saved dataset_info.json")
