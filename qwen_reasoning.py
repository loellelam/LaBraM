# qwen_reasoning.py
# Takes LaBraM's prediction for a single EEG recording and asks Qwen
# to generate clinical reasoning explaining the result.

import torch
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from timm.models import create_model
from data_processor.AD import ADDataset
import modeling_finetune
import utils

# ---- config ----
LABRAM_CHECKPOINT  = "./checkpoints/finetune_ad/checkpoint-best.pth"
QWEN_MODEL_NAME    = "Qwen/Qwen2.5-0.5B-Instruct" # small model
# QWEN_MODEL_NAME    = "Qwen/Qwen2.5-3B-Instruct" # medium model
# QWEN_MODEL_NAME    = "Qwen/Qwen2.5-7B-Instruct" # large model
DATA_PATH          = "./labram_data"
CH_NAMES_PATH      = "./labram_data/channel_names.json"
NB_CLASSES         = 1
LABEL_NAMES        = {0: "Non-Alzheimer's Disease (Non-AD)", 1: "Alzheimer's Disease (AD)"}
# ----------------

# =====================
# 1. Load LaBraM
# =====================
def load_labram(checkpoint_path, nb_classes):
    model = create_model(
        "labram_base_patch200_200",
        pretrained=False,
        num_classes=nb_classes,
        use_mean_pooling=True,
        use_rel_pos_bias=False,
        use_abs_pos_emb=True,
        init_values=0.1,
        qkv_bias=False,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


# =====================
# 2. Run LaBraM inference on one subject's windows
# =====================
def run_labram_inference(model, X, ch_names):
    """
    X: numpy array of shape (n_windows, n_channels, n_times)
    Returns:
        predicted_label: int (majority vote across windows)
        confidence: float (mean probability of predicted class)
        per_class_probs: dict with mean probability per class
        window_votes: list of per-window predictions
    """
    # Use the same function that was used during finetuning
    input_chans = utils.get_input_chans(ch_names)
    print(f"input_chans: {input_chans}")

    all_probs = []
    with torch.no_grad():
        for i in range(len(X)):
            x = torch.tensor(X[i]).float().unsqueeze(0)  # (1, n_ch, 800)
            x = x.reshape(x.shape[0], x.shape[1], -1, 200)  # (1, n_ch, 4, 200)
            logits = model(x, input_chans)
            # for nb_classes=1, logits is shape (1, 1) — use sigmoid
            prob_ad = torch.sigmoid(logits).squeeze().item()
            probs = np.array([1 - prob_ad, prob_ad])  # [non-AD, AD]
            # probs = torch.softmax(logits, dim=-1).squeeze().numpy()
            all_probs.append(probs)
    
    all_probs = np.array(all_probs)
    mean_probs = all_probs.mean(axis=0)
    predicted_label = int(mean_probs.argmax())
    confidence = float(mean_probs[predicted_label])
    window_votes = all_probs.argmax(axis=1).tolist()

    per_class_probs = {
        LABEL_NAMES[0]: f"{mean_probs[0]*100:.1f}%",
        LABEL_NAMES[1]: f"{mean_probs[1]*100:.1f}%",
    }

    return predicted_label, confidence, per_class_probs, window_votes

# =====================
# 3. Extract interpretable EEG features for Qwen context
# =====================
def extract_eeg_summary(X, ch_names, sfreq=200):
    """
    Compute simple band power features to give Qwen grounding.
    Returns a human-readable string summary.
    """
    bands = {
        "delta (0.5-4 Hz)":   (0.5, 4),
        "theta (4-8 Hz)":     (4, 8),
        "alpha (8-13 Hz)":    (8, 13),
        "beta (13-30 Hz)":    (13, 30),
    }

    # Average across all windows
    X_mean = X.mean(axis=0)  # (n_channels, n_times)

    band_powers = {}
    for band_name, (flo, fhi) in bands.items():
        # Use FFT to compute band power
        freqs = np.fft.rfftfreq(X_mean.shape[-1], d=1.0/sfreq)
        fft_vals = np.abs(np.fft.rfft(X_mean, axis=-1)) ** 2
        mask = (freqs >= flo) & (freqs <= fhi)
        band_powers[band_name] = float(fft_vals[:, mask].mean())

    # Normalize to relative power
    total = sum(band_powers.values())
    rel_powers = {k: v/total for k, v in band_powers.items()}

    # Key AD/FTD biomarkers
    theta_alpha_ratio = band_powers["theta (4-8 Hz)"] / (band_powers["alpha (8-13 Hz)"] + 1e-10)

    lines = ["EEG band power summary (relative):"]
    for band, power in rel_powers.items():
        lines.append(f"  {band}: {power*100:.1f}%")
    lines.append(f"  Theta/alpha ratio: {theta_alpha_ratio:.3f} (elevated in AD/FTD)")
    lines.append(f"  Number of EEG windows analyzed: {len(X)}")
    lines.append(f"  Channels: {', '.join(ch_names[:5])}... ({len(ch_names)} total)")

    return "\n".join(lines)


# =====================
# 4. Build Qwen prompt
# =====================
def build_prompt(predicted_label, confidence, per_class_probs, window_votes, eeg_summary):
    label_name = LABEL_NAMES[predicted_label]
    consistency = window_votes.count(predicted_label) / len(window_votes) * 100

    prompt = f"""You are a clinical neurologist specializing in dementia and EEG analysis. 
You have been given the results of an AI analysis of a patient's EEG recording.

## AI Model Results
- Primary diagnosis: {label_name}
- Model confidence: {confidence*100:.1f}%
- Prediction consistency across EEG windows: {consistency:.1f}% of windows agreed
- Per-class probabilities:
{chr(10).join(f'  - {cls}: {prob}' for cls, prob in per_class_probs.items())}

## EEG Signal Analysis
{eeg_summary}

## Your Task
Based on the AI model's prediction and the EEG features above, provide:

1. **Clinical interpretation**: What does this EEG pattern suggest clinically? 
2. **Supporting evidence**: Which specific EEG features support this diagnosis?
3. **Differential diagnosis**: Why was {LABEL_NAMES[(predicted_label+1) % 2]} less likely?
4. **Confidence assessment**: Is the model's confidence level appropriate given the EEG features?
5. **Clinical recommendations**: What follow-up steps would you recommend?

Be specific about EEG biomarkers (delta/theta slowing, alpha suppression, etc.) and 
ground your reasoning in established clinical knowledge about AD and healthy aging.
Keep your response focused and clinically useful."""

    return prompt


# =====================
# 5. Run Qwen
# =====================
def run_qwen(prompt, model_name=QWEN_MODEL_NAME):
    print(f"Loading Qwen model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    qwen = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    messages = [
        {"role": "system", "content": "You are an expert clinical neurologist specializing in EEG-based dementia diagnosis."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(qwen.device)

    with torch.no_grad():
        outputs = qwen.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.3,    # low temperature = more focused clinical reasoning
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[-1]:],
        skip_special_tokens=True
    )
    return response


# =====================
# 6. Main — run on one test subject
# =====================
def main():
    with open(CH_NAMES_PATH) as f:
        ch_names = json.load(f)

    # Load LaBraM
    print("Loading LaBraM...")
    labram = load_labram(LABRAM_CHECKPOINT, NB_CLASSES)

    # Load test data — run on first subject's windows as demo
    # In practice you'd select a specific subject
    import h5py
    with h5py.File(f"{DATA_PATH}/test.h5") as f:
        X_all = f["X"][:]
        y_all = f["y"][:]

    # Pick first 50 windows (one subject worth) for demo
    # Ideally filter by subject ID using split_info.json
    X_sample = X_all[:50]
    true_label = int(y_all[0])

    print(f"True label: {LABEL_NAMES[true_label]}")
    print("Running LaBraM inference...")

    pred_label, confidence, per_class_probs, window_votes = run_labram_inference(
        labram, X_sample, ch_names
    )

    print(f"Predicted: {LABEL_NAMES[pred_label]} ({confidence*100:.1f}% confidence)")

    eeg_summary = extract_eeg_summary(X_sample, ch_names)
    print("\nEEG Summary:")
    print(eeg_summary)

    prompt = build_prompt(pred_label, confidence, per_class_probs, window_votes, eeg_summary)

    print("\nSending to Qwen for clinical reasoning...")
    reasoning = run_qwen(prompt)

    print("\n" + "="*50)
    print("CLINICAL REASONING (Qwen)")
    print("="*50)
    print(reasoning)

    # Save output
    output = {
        "true_label": LABEL_NAMES[true_label],
        "predicted_label": LABEL_NAMES[pred_label],
        "confidence": confidence,
        "per_class_probs": per_class_probs,
        "window_consistency": f"{window_votes.count(pred_label)/len(window_votes)*100:.1f}%",
        "eeg_summary": eeg_summary,
        "clinical_reasoning": reasoning,
    }
    with open("reasoning_output.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nSaved to reasoning_output.json")


if __name__ == "__main__":
    main()