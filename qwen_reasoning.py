# qwen_reasoning.py
# Takes LaBraM's prediction for a single EEG recording and asks Qwen
# to generate clinical reasoning explaining the result.

import torch
import json
import numpy as np
# from transformers import AutoModelForCausalLM, AutoTokenizer
import ollama
from timm.models import create_model
from data_processor.AD import ADDataset
import modeling_finetune
from scipy.signal import welch
from scipy.signal import coherence
import utils

# ---- config ----
LABRAM_CHECKPOINT  = "./checkpoints/finetune_ad/checkpoint-best.pth"
# QWEN_MODEL_NAME    = "Qwen/Qwen2.5-0.5B-Instruct" # small model
# QWEN_MODEL_NAME    = "Qwen/Qwen2.5-3B-Instruct" # medium model
# QWEN_MODEL_NAME    = "Qwen/Qwen2.5-7B-Instruct" # large model
MAX_NEW_TOKENS      = 3000
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
# def extract_eeg_summary(X, ch_names, sfreq=200):
#     """
#     Compute interpretable features to give Qwen grounding.
#     Returns a human-readable string summary.
#     """
#     bands = {
#         "delta (0.5-4 Hz)":   (0.5, 4),
#         "theta (4-8 Hz)":     (4, 8),
#         "alpha (8-13 Hz)":    (8, 13),
#         "beta (13-30 Hz)":    (13, 30),
#     }

#     # Average across all windows
#     # X_mean = X.mean(axis=0)  # (n_channels, n_times)
#     all_psds = []
#     for window in X:
#         freqs, psd = welch(window, fs=sfreq, nperseg=256, axis=-1)
#         all_psds.append(psd)
#     X_mean = np.mean(all_psds, axis=0)

#     freqs, psd = welch(
#         X_mean,
#         fs=sfreq,
#         nperseg=min(512, X_mean.shape[-1]),
#         axis=-1
#     )

#     band_powers = {}
#     for band_name, (flo, fhi) in bands.items():
#         mask = (freqs >= flo) & (freqs <= fhi)
#         band_powers[band_name] = float(psd[:, mask].mean())
#         # Use FFT to compute band power
#         # freqs = np.fft.rfftfreq(X_mean.shape[-1], d=1.0/sfreq)
#         # fft_vals = np.abs(np.fft.rfft(X_mean, axis=-1)) ** 2
#         # mask = (freqs >= flo) & (freqs <= fhi)
#         # band_powers[band_name] = float(fft_vals[:, mask].mean())

#     # Normalize to relative power
#     total = sum(band_powers.values())
#     rel_powers = {k: v/total for k, v in band_powers.items()}

#     # Posterior Alpha Peak Frequency (PAF)
#     POSTERIOR_CHS = ["P3", "P4", "PZ", "O1", "O2"]
#     posterior_idx = [
#         i for i, ch in enumerate(ch_names)
#         if ch in POSTERIOR_CHS
#     ]
#     posterior_signal = X_mean[posterior_idx].mean(axis=0)
#     freqs = np.fft.rfftfreq(len(posterior_signal), d=1/sfreq)
#     psd = np.abs(np.fft.rfft(posterior_signal)) ** 2
#     alpha_mask = (freqs >= 8) & (freqs <= 13)
#     alpha_freqs = freqs[alpha_mask]
#     alpha_psd = psd[alpha_mask]
#     peak_alpha_freq = float(alpha_freqs[np.argmax(alpha_psd)])

#     # Posterior Alpha Relative Power
#     # alpha_power = alpha_psd.mean()
#     # total_power = psd.mean()
#     # posterior_alpha_relative_power = alpha_power / total_power
#     broadband_mask = (freqs >= 0.5) & (freqs <= 30)
#     alpha_power = np.sum(alpha_psd)
#     total_power = np.sum(psd[broadband_mask])
#     posterior_alpha_relative_power = alpha_power / (total_power + 1e-10)

#     # Posterior Alpha Coherence
#     # corr_matrix = np.corrcoef(X_mean[posterior_idx])
#     # upper_triangle = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
#     # posterior_coherence = float(np.mean(np.abs(upper_triangle)))
#     coh_vals = []
#     for i in range(len(posterior_idx)):
#         for j in range(i + 1, len(posterior_idx)):

#             f, cxy = coherence(
#                 X_mean[posterior_idx[i]],
#                 X_mean[posterior_idx[j]],
#                 fs=sfreq,
#                 nperseg=256
#             )

#             alpha_mask = (f >= 8) & (f <= 13)

#             coh_vals.append(np.mean(cxy[alpha_mask]))
#     posterior_coherence = float(np.mean(coh_vals))

#     # Spectral Slowing Index
#     slowing_index = (
#         band_powers["delta (0.5-4 Hz)"] +
#         band_powers["theta (4-8 Hz)"]
#     ) / (
#         band_powers["alpha (8-13 Hz)"] +
#         band_powers["beta (13-30 Hz)"]
#     )

#     # Theta/alpha ratio
#     theta_alpha_ratio = band_powers["theta (4-8 Hz)"] / (band_powers["alpha (8-13 Hz)"] + 1e-10)

#     # Key AD/FTD biomarkers summary
#     lines = []
#     lines.append(f"Number of EEG windows analyzed: {len(X)}")
#     lines.append(f"Channels: {', '.join(ch_names[:5])}... ({len(ch_names)} total)")

#     lines.append("EEG band power summary (relative):")
#     for band, power in rel_powers.items():
#         pct = power * 100

#         # Typical adult resting EEG ranges
#         ref = {
#             "delta (0.5-4 Hz)": "Normal: 5–20% | AD often: >20–30%",
#             "theta (4-8 Hz)": "Normal: 10–25% | AD often: >25–35%",
#             "alpha (8-13 Hz)": "Normal: 25–45% | AD often: <20–25%",
#             "beta (13-30 Hz)":  "Normal: 10–25% | AD often: <10–15%",
#         }.get(band, "No reference")

#         lines.append(f"  {band}: {pct:.1f}% ({ref})")

#     # Posterior Alpha Peak Frequency (PAPF)
#     # Healthy older adults: ~9–11 Hz
#     # AD commonly: <8–9 Hz
#     lines.append(
#         f"Posterior Alpha Peak Frequency: {peak_alpha_freq:.1f} Hz "
#         "(Normal: 9–11 Hz | AD often: <8–9 Hz)"
#     )

#     # Posterior alpha relative power
#     # Healthy: stronger posterior alpha
#     # AD: reduced posterior alpha
#     lines.append(
#         f"Posterior Alpha Relative Power: "
#         f"{posterior_alpha_relative_power*100:.1f}% "
#         "(Normal: >20–30% | AD often: <15–20%)"
#     )

#     # Coherence
#     # AD often shows reduced posterior coherence/connectivity
#     lines.append(
#         f"Posterior Alpha Coherence: {posterior_coherence*100:.1f}% "
#         "(Normal: >50–70% | AD often: <40–50%)"
#     )

#     # Spectral slowing index
#     # Higher = more slowing (delta/theta dominance)
#     lines.append(
#         f"Spectral Slowing Index: {slowing_index:.3f} "
#         "(Normal: <1.0 | AD often: >1.2–1.5)"
#     )

#     # Theta/alpha ratio
#     # Elevated in AD due to increased theta + reduced alpha
#     lines.append(
#         f"Theta/alpha ratio: {theta_alpha_ratio:.3f} "
#         "(Normal: <0.8–1.0 | AD often: >1.2)"
#     )

#     return "\n".join(lines)
def extract_eeg_summary(X, ch_names, sfreq=200):
    """
    Compute clinically interpretable EEG biomarkers for Qwen.
    X shape: (n_windows, n_channels, n_times)
    """

    # ---------------------------------------------------
    # Channel groups
    # ---------------------------------------------------

    POSTERIOR_CHS = ["P3", "P4", "PZ", "O1", "O2"]

    posterior_idx = [
        i for i, ch in enumerate(ch_names)
        if ch.upper() in POSTERIOR_CHS
    ]

    # ---------------------------------------------------
    # Frequency bands
    # ---------------------------------------------------

    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
    }

    # ---------------------------------------------------
    # Aggregate PSD across ALL windows/channels
    # ---------------------------------------------------

    band_abs_power = {k: [] for k in bands}

    posterior_alpha_powers = []
    posterior_peak_freqs = []

    coherence_vals = []

    for win in X:

        # ---------------------------------------------
        # Per-channel PSD
        # ---------------------------------------------

        for ch_i in range(win.shape[0]):

            signal = win[ch_i]

            freqs, psd = welch(
                signal,
                fs=sfreq,
                nperseg=min(512, len(signal))
            )

            total_power = np.trapz(psd, freqs)

            for band_name, (f_lo, f_hi) in bands.items():

                mask = (freqs >= f_lo) & (freqs <= f_hi)

                band_power = np.trapz(psd[mask], freqs[mask])

                band_abs_power[band_name].append(band_power)

        # ---------------------------------------------
        # Posterior alpha peak frequency
        # ---------------------------------------------

        posterior_signal = win[posterior_idx].mean(axis=0)

        freqs, psd = welch(
            posterior_signal,
            fs=sfreq,
            nperseg=min(512, len(posterior_signal))
        )

        alpha_mask = (freqs >= 8) & (freqs <= 13)

        alpha_freqs = freqs[alpha_mask]
        alpha_psd = psd[alpha_mask]

        peak_freq = alpha_freqs[np.argmax(alpha_psd)]
        posterior_peak_freqs.append(float(peak_freq))

        alpha_power = np.trapz(alpha_psd, alpha_freqs)
        total_power = np.trapz(psd, freqs)

        posterior_alpha_powers.append(
            alpha_power / (total_power + 1e-12)
        )

        # ---------------------------------------------
        # Posterior alpha coherence
        # ---------------------------------------------

        posterior_pairs = []

        for i in range(len(posterior_idx)):
            for j in range(i + 1, len(posterior_idx)):

                sig1 = win[posterior_idx[i]]
                sig2 = win[posterior_idx[j]]

                f_coh, coh = coherence(
                    sig1,
                    sig2,
                    fs=sfreq,
                    nperseg=min(256, len(sig1))
                )

                alpha_mask = (f_coh >= 8) & (f_coh <= 13)

                alpha_coh = np.mean(coh[alpha_mask])

                posterior_pairs.append(alpha_coh)

        if posterior_pairs:
            coherence_vals.append(np.mean(posterior_pairs))

    # ---------------------------------------------------
    # Mean band powers
    # ---------------------------------------------------

    mean_band_powers = {
        k: np.mean(v)
        for k, v in band_abs_power.items()
    }

    total = sum(mean_band_powers.values())

    rel_powers = {
        k: v / total
        for k, v in mean_band_powers.items()
    }

    # ---------------------------------------------------
    # Biomarkers
    # ---------------------------------------------------

    theta_alpha_ratio = (
        mean_band_powers["theta"] /
        (mean_band_powers["alpha"] + 1e-12)
    )

    slowing_index = (
        mean_band_powers["delta"] +
        mean_band_powers["theta"]
    ) / (
        mean_band_powers["alpha"] +
        mean_band_powers["beta"] +
        1e-12
    )

    posterior_alpha_peak = np.mean(posterior_peak_freqs)

    posterior_alpha_relative_power = np.mean(
        posterior_alpha_powers
    )

    posterior_alpha_coherence = np.mean(
        coherence_vals
    )

    # ---------------------------------------------------
    # Build human-readable summary
    # ---------------------------------------------------

    lines = []

    lines.append(
        f"Windows analyzed: {len(X)}"
    )

    lines.append(
        f"Channels: {len(ch_names)}"
    )

    lines.append("")

    lines.append("Relative Band Power:")

    for band, val in rel_powers.items():
        lines.append(
            f"  {band}: {val:.3f}"
        )

    lines.append("")

    lines.append(
        f"Posterior alpha peak frequency: "
        f"{posterior_alpha_peak:.2f} Hz"
    )

    lines.append(
        f"Posterior alpha relative power: "
        f"{posterior_alpha_relative_power:.3f}"
    )

    lines.append(
        f"Posterior alpha coherence: "
        f"{posterior_alpha_coherence:.3f}"
    )

    lines.append(
        f"Theta/alpha ratio: "
        f"{theta_alpha_ratio:.3f}"
    )

    lines.append(
        f"Spectral slowing index: "
        f"{slowing_index:.3f}"
    )

    return "\n".join(lines)

# =====================
# 4. Build Qwen prompt
# =====================
# def build_prompt(predicted_label, confidence, per_class_probs, window_votes, eeg_summary):
#     label_name = LABEL_NAMES[predicted_label]
#     consistency = window_votes.count(predicted_label) / len(window_votes) * 100

#     prompt = f"""You are a clinical neurologist specializing in dementia and EEG analysis. 
# You have been given the results of an AI analysis of a patient's EEG recording.

# ## AI Model Results
# - Primary diagnosis: {label_name}
# - Model confidence: {confidence*100:.1f}%
# - Prediction consistency across EEG windows: {consistency:.1f}% of windows agreed
# - Per-class probabilities:
# {chr(10).join(f'  - {cls}: {prob}' for cls, prob in per_class_probs.items())}

# ## EEG Signal Analysis
# {eeg_summary}

# ## Your Task
# Based on the AI model's prediction and the EEG features above, provide:

# 1. **Clinical interpretation**: What does this EEG pattern suggest clinically? 
# 2. **Supporting evidence**: Which specific EEG features support this diagnosis?
# 3. **Differential diagnosis**: Why was {LABEL_NAMES[(predicted_label+1) % 2]} less likely?
# 4. **Confidence assessment**: Is the model's confidence level appropriate given the EEG features?
# 5. **Clinical recommendations**: What follow-up steps would you recommend?

# Be specific about EEG biomarkers (delta/theta slowing, alpha suppression, etc.) and 
# ground your reasoning in established clinical knowledge about AD and healthy aging.
# Keep your response focused and clinically useful."""

#     return prompt

# ==============================================================================================
# def build_prompt(per_class_probs, confidence, eeg_summary):
#     prompt = f"""
# You are an expert neurologist specializing in EEG biomarkers of Alzheimer's disease. Given quantitative EEG features extracted from a patient recording, produce a structured differential assessment.

# Write in plain human-readable prose only, no markdown language.

# The AI foundation EEG model (LaBraM) produced:
# - Non-AD probability: {per_class_probs[LABEL_NAMES[0]]}
# - AD probability: {per_class_probs[LABEL_NAMES[1]]}
# - Note: The AI model may be incorrect. Use the EEG evidence independently and critically.

# EEG BIOMARKER SUMMARY
# {eeg_summary}

# YOUR TASK:

# Write a structured clinical reasoning report with the following sections.

# 1. Objective Findings

# Specify:
# - EEG biomarker summary
# - LaBraM model prediction

# Do NOT diagnose yet.

# 2. Differential Diagnosis

# Generate multiple hypotheses including:
# - Alzheimer's disease
# - Non-AD / healthy aging
# - Other cognitive impairment if relevant

# For EACH hypothesis provide:
# - Supporting EEG evidence
# - Contradicting EEG evidence
# - Preliminary likelihood

# Use bullet points.

# Example format:

# Hypothesis 1: Alzheimer's Disease
#     Supporting EEG Evidence:
#     - Posterior alpha slowing
#     - Increased theta activity
#     Contradicting Evidence:
#     - Preserved posterior alpha power
#     Preliminary Likelihood: Moderate

# 3. Uncertainty and Confidence

# Discuss:
# - Overall confidence of your diagnosis and justification
# - Reliability of biomarkers
# - What additional data would change your prediction or confidence

# 4. Recommended Next Steps

# Suggest:
# - Additional EEG analysis if needed
# - Cognitive testing
# - MRI/PET correlation
# - Follow-up recommendations

# Be concise but clinically rigorous.
# """
#     return prompt

# ==========================================================================
def build_prompt(predicted_label, confidence, per_class_probs, window_votes, eeg_summary):
    label_name  = LABEL_NAMES[predicted_label]
    consistency = window_votes.count(predicted_label) / len(window_votes)
    ad_prob     = float(per_class_probs[LABEL_NAMES[1]].strip("%")) / 100
    nonad_prob  = float(per_class_probs[LABEL_NAMES[0]].strip("%")) / 100

    prompt = f"""You are a clinical neurologist writing a structured EEG analysis report.

CRITICAL RULES — violating any of these makes the report unusable:
- Use ONLY plain text. No asterisks, no pound signs, no markdown of any kind.
- Use ONLY the numbers provided in the INPUT DATA section below. Do NOT invent any values.
- Copy the section headers and divider lines EXACTLY as shown in OUTPUT FORMAT.
- Fill in [bracketed placeholders] with your clinical reasoning.
- Use dashes (-) for list items. No other bullet symbols.
- Do not add any sections not shown in the format.

============================================================
INPUT DATA
============================================================

LaBraM Foundation Model Output:
  Non-AD probability : {nonad_prob:.3f}
  AD probability     : {ad_prob:.3f}
  Windows analyzed   : {window_votes.__len__()}
  Window consistency : {consistency*100:.1f}% of windows predicted {label_name}

EEG Biomarker Measurements:
{eeg_summary}

============================================================
OUTPUT FORMAT — reproduce this exactly, replacing placeholders
============================================================

============================================================
  1. OBJECTIVE FINDINGS
============================================================

  LaBraM Foundation Model
  ----------------------------------------
  Windows analyzed   : {window_votes.__len__()}
  Window consistency : {consistency*100:.1f}% of windows predicted {label_name}
  Non-AD probability : {nonad_prob*100:.1f}%
  AD probability     : {ad_prob*100:.1f}%

  Key AD Biomarkers
  ----------------------------------------
  - [Copy each biomarker line from the input data exactly, then add one sentence of clinical interpretation below each one. Do not change any numbers.]

============================================================
  2. DIFFERENTIAL DIAGNOSIS
============================================================

  Impression: [One sentence overall impression using only the data provided]

  #1  Alzheimer Disease  [Confidence: XX%]
      [Two sentences of reasoning citing specific numbers from the input data]
      Supporting:
        - [biomarker from input data and why it supports AD]
        - [biomarker from input data and why it supports AD]
      Against:
        - [biomarker from input data that argues against AD]

  #2  Non-AD / Healthy aging  [Confidence: XX%]
      [Two sentences of reasoning citing specific numbers from the input data]
      Supporting:
        - [biomarker from input data and why it supports non-AD]
      Against:
        - [biomarker from input data that argues against non-AD]

============================================================
  3. UNCERTAINTY AND CONFIDENCE
============================================================

  Overall confidence : XX%
  Justification: [One sentence]

  Limiting factors:
    - [factor specific to this recording]
    - [factor specific to this recording]

  What would change this assessment:
    - [specific condition or additional data]
    - [specific condition or additional data]

============================================================
  4. NEXT STEPS
============================================================

  1. [clinical recommendation]
  2. [clinical recommendation]
  3. [clinical recommendation]
  4. [clinical recommendation]

============================================================
"""
    return prompt


# =====================
# 5. Run Qwen
# =====================
# def run_qwen(prompt, model_name=QWEN_MODEL_NAME):
#     print(f"Loading Qwen model: {model_name}")
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     qwen = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
#         device_map="auto",
#     )

#     messages = [
#         {
#             "role": "system",
#             "content": (
#                 "You are a clinical neurologist writing structured EEG reports. "
#                 "You write in plain text only — no markdown, no asterisks, no pound signs. "
#                 "You never invent data. You follow the output format exactly as specified."
#             )
#         },
#         {"role": "user", "content": prompt}
#     ]

#     text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     inputs = tokenizer([text], return_tensors="pt").to(qwen.device)

#     with torch.no_grad():
#         outputs = qwen.generate(
#             **inputs,
#             max_new_tokens=MAX_NEW_TOKENS,
#             temperature=0.3,    # low temperature = more focused clinical reasoning
#             do_sample=True,
#             pad_token_id=tokenizer.eos_token_id,
#         )

#     response = tokenizer.decode(
#         outputs[0][inputs.input_ids.shape[-1]:],
#         skip_special_tokens=True
#     )
#     return response
def run_qwen(prompt):
    try:
        response = ollama.chat(
            model="qwen3:8b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert clinical neurophysiologist "
                        "specializing in EEG biomarkers of dementia."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            options={
                "temperature": 0.3,
                "num_predict": MAX_NEW_TOKENS,
            },
            think=False
        )

        print("FULL RESPONSE:")
        print(response)
        print("END FULL RESPONSE")

        return response["message"]["content"]
    
    except Exception as e:
        print("OLLAMA ERROR:")
        print(e)
        return "ERROR"

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
    X_sample = X_all # X_all[:50]
    true_label = int(y_all[0])

    print(f"True label: {LABEL_NAMES[true_label]}")
    print("Running LaBraM inference...")

    pred_label, confidence, per_class_probs, window_votes = run_labram_inference(
        labram, X_sample, ch_names
    )

    print(f"LaBraM Prediction: {LABEL_NAMES[pred_label]} ({confidence*100:.1f}% confidence)")

    eeg_summary = extract_eeg_summary(X_sample, ch_names)
    print("\nEEG Biomarker Summary:")
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