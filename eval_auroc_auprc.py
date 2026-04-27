# Written by Sang

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize
from einops import rearrange

# --- imports from your project ---
from custom_labram_dataset import LabramH5Dataset
import modeling_finetune
from timm.models import create_model
import utils

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- paths ---
h5_path = "/mnt/c/Users/sangt/Downloads/OpenNeuro/Dataset/final_dataset.h5"
split_npz = "/mnt/c/Users/sangt/Downloads/OpenNeuro/Dataset/window_splits.npz"
ckpt_path = "./out_openneuro3_run3/checkpoint-best.pth"

# --- dataset ---
test_dataset = LabramH5Dataset(h5_path, split_npz, split="test")
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)

ch_names = test_dataset.channels
input_chans = utils.get_input_chans(ch_names)

# --- model ---
model = create_model(
    "labram_base_patch200_200",
    pretrained=False,
    num_classes=3,
    drop_rate=0.0,
    drop_path_rate=0.1,
    attn_drop_rate=0.0,
    use_mean_pooling=True,
    init_scale=0.001,
    use_rel_pos_bias=True,
    use_abs_pos_emb=True,
    init_values=0.1,
    qkv_bias=True,
)

checkpoint = torch.load(ckpt_path, map_location="cpu")
state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
msg = model.load_state_dict(state_dict, strict=False)
print("load_state_dict:", msg)

model.to(DEVICE)
model.eval()

# --- collect outputs ---
all_probs = []
all_labels = []

with torch.no_grad():
    for x, y in tqdm(test_loader):
        x = x.float().to(DEVICE) / 100
        x = rearrange(x, 'B N (A T) -> B N A T', T=200)

        outputs = model(x, input_chans=input_chans)
        probs = torch.softmax(outputs, dim=1)

        all_probs.append(probs.cpu().numpy())
        all_labels.append(y.numpy())

# --- stack ---
all_probs = np.concatenate(all_probs, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

# --- binarize labels ---
y_true = label_binarize(all_labels, classes=[0, 1, 2])

# --- AUROC ---
roc_auc = roc_auc_score(y_true, all_probs, average="macro", multi_class="ovr")

# --- AUPRC ---
pr_auc = average_precision_score(y_true, all_probs, average="macro")

print("\n=== Evaluation ===")
print(f"Macro AUROC: {roc_auc:.4f}")
print(f"Macro AUPRC: {pr_auc:.4f}")