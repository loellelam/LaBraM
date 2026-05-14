import os
import json
import torch
import numpy as np
from pathlib import Path
from einops import rearrange
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from timm.models import create_model

# workspace helpers
from qwen_reasoning import load_labram  # loads model + checkpoint as in repo
from data_processor.AD import ADDataset   # dataset used in run_class_finetuning.py
import utils  # for get_input_chans helper

# --- CONFIG: adjust these paths ---
CKPT_PATH = "./checkpoints/finetune_ad/checkpoint-best.pth"
TEST_H5 = "./labram_data_adftd/test.h5"
CH_NAMES = "./labram_data_adftd/channel_names.json"
NB_CLASSES = 1  # 1 for binary (BCEWithLogits) as used in repo
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
THRESHOLD = 0.5
# --- end config ---

def load_channel_names(path):
    with open(path) as f:
        return json.load(f)

def inference_and_metrics():
    ch_names = load_channel_names(CH_NAMES)
    model = load_labram(CKPT_PATH, NB_CLASSES)
    model.to(DEVICE)
    model.eval()

    test_ds = ADDataset(TEST_H5, ch_names)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X, y in test_loader:
            X = X.float().to(DEVICE) / 100.0
            # repo uses shape (B, N, (A*T)) and then rearranges to (B, N, A, T)
            X = rearrange(X, 'B N (A T) -> B N A T', T=200)
            logits = model(X, utils.get_input_chans(ch_names)) if hasattr(model, '__call__') else model(X)
            if NB_CLASSES == 1:
                probs = torch.sigmoid(logits).cpu().numpy().squeeze()
            else:
                probs = torch.softmax(logits, dim=1).cpu().numpy()  # shape (B, C)
            all_probs.append(probs)
            all_labels.append(y.numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0).astype(int)

    if NB_CLASSES == 1:
        # per-sample AD probability
        prob_ad = all_probs.reshape(-1)
        preds = (prob_ad >= THRESHOLD).astype(int)
        y_true = all_labels
    else:
        preds = all_probs.argmax(axis=1)
        # convert to binary per-class if needed
        y_true = all_labels

    # compute metrics (binary case)
    if NB_CLASSES == 1:
        precision = precision_score(y_true, preds, zero_division=0)
        recall = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        # for multiclass, use macro averages (adjust if you want per-class)
        precision = precision_score(y_true, preds, average='macro', zero_division=0)
        recall = recall_score(y_true, preds, average='macro', zero_division=0)
        f1 = f1_score(y_true, preds, average='macro', zero_division=0)
        # specificity per-class is more involved for multiclass; you can compute from confusion matrix per class
        cm = confusion_matrix(y_true, preds)
        # compute macro-specificity (avg of TN/(TN+FP) per class)
        specificities = []
        for i in range(cm.shape[0]):
            tp_i = cm[i, i]
            fp_i = cm[:, i].sum() - tp_i
            fn_i = cm[i, :].sum() - tp_i
            tn_i = cm.sum() - (tp_i + fp_i + fn_i)
            spec_i = tn_i / (tn_i + fp_i) if (tn_i + fp_i) > 0 else 0.0
            specificities.append(spec_i)
        specificity = float(np.mean(specificities))

    print("=== Metrics ===")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1 score:    {f1:.4f}")

if __name__ == "__main__":
    inference_and_metrics()
