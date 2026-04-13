import h5py, numpy as np, json
from pathlib import Path

# Check class balance in each split
print("Class balance in each split:")
label_map = {0: "AD", 1: "FTD", 2: "Control"}
for split in ['train', 'val', 'test']:
    with h5py.File(f'labram_data/{split}.h5', 'r') as f:
        y = f['y'][:]
        total = len(y)

        print(f"{split}: {total} windows")

        unique, counts = np.unique(y, return_counts=True)
        count_dict = dict(zip(unique, counts))

        for label_id, label_name in label_map.items():
            count = count_dict.get(label_id, 0)
            pct = 100 * count / total if total > 0 else 0
            print(f"  {label_name} = {count} ({pct:.1f}%)")

# Check the manifest to see how many unique subjects were processed
# with open('output/manifest.json') as f:
#     manifest = json.load(f)
# print(f'\nTotal subjects processed: {len(manifest["results"])}')
# for r in manifest['results']:
#     print(f'  {Path(r["output_h5"]).stem}  ->  {r["n_windows"]} windows')

# Collect all epoch results
epochs = []
with open('./checkpoints/finetune_ad/log.txt') as f:
    print('epoch | train_loss | val_acc | test_acc')
    for line in f:
        data = json.loads(line)
        epochs.append(data)
        print(f'  {data["epoch"]:>2}  |  {data.get("train_loss",0):.4f}      |  {data.get("val_accuracy",0):.4f}   |  {data.get("test_accuracy",0):.4f}')
# Find best epoch
best = max(epochs, key=lambda x: x['val_accuracy'])
# best = epochs[-1]
print(f'Best epoch: {best["epoch"]}')
print(f'  train_class_accuracy:  {best["train_class_acc"]:.4f}')
print()
print(f'  val_accuracy:          {best["val_accuracy"]:.4f}')
print(f'  val_balanced_accuracy: {best["val_balanced_accuracy"]:.4f}')
print(f'  val_cohen_kappa:       {best["val_cohen_kappa"]:.4f}')
print(f'  val_f1_weighted:       {best["val_f1_weighted"]:.4f}')
print()
print(f'  test_accuracy:          {best["test_accuracy"]:.4f}')
print(f'  test_balanced_accuracy: {best["test_balanced_accuracy"]:.4f}')
print(f'  test_cohen_kappa:       {best["test_cohen_kappa"]:.4f}')
print(f'  test_f1_weighted:       {best["test_f1_weighted"]:.4f}')