import h5py, numpy as np, json
from pathlib import Path

# Check class balance in each split
for split in ['train', 'val', 'test']:
    with h5py.File(f'labram_data/{split}.h5') as f:
        y = f['y'][:]
        total = len(y)
        ad = y.sum()
        ctrl = (y == 0).sum()
        print(f'{split}: {total} windows | AD={ad} ({100*ad/total:.1f}%) | Control={ctrl} ({100*ctrl/total:.1f}%)')

# Check the manifest to see how many unique subjects were processed
with open('output/manifest.json') as f:
    manifest = json.load(f)
print(f'\nTotal subjects processed: {len(manifest["results"])}')
for r in manifest['results']:
    print(f'  {Path(r["output_h5"]).stem}  ->  {r["n_windows"]} windows')

import json
with open('./checkpoints/finetune_ad/log.txt') as f:
    lines = f.readlines()
print('epoch | train_loss | val_acc | test_acc')
for line in lines:
    d = json.loads(line)
    print(f'  {d["epoch"]:>2}  |  {d.get("train_loss",0):.4f}      |  {d.get("val_accuracy",0):.4f}   |  {d.get("test_accuracy",0):.4f}')
