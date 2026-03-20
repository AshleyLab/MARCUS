# Model Card: MARCUS-ECG

## Model Overview

| Field             | Value                                                                 |
|-------------------|-----------------------------------------------------------------------|
| Model ID          | `stanford-cardiac-ai/MARCUS-ECG`                                      |
| Base model        | `Qwen/Qwen2.5-VL-3B-Instruct`                                         |
| Parameters        | 3 billion                                                             |
| Modality          | Electrocardiography (ECG)                                             |
| Input             | 12-lead ECG rendered as hospital-style PNG                            |
| Output            | Free-text clinical interpretation                                     |
| Training method   | Stage 1 encoder pretraining + Stage 2 SFT                            |
| License           | MIT (model weights: Qwen License)                                     |
| HuggingFace       | [stanford-cardiac-ai/MARCUS-ECG](https://huggingface.co/stanford-cardiac-ai/MARCUS-ECG) |

---

## Description

MARCUS-ECG is a 3-billion-parameter vision-language model specialized for 12-lead electrocardiogram interpretation. It is one of three expert components of the MARCUS cardiac AI system, described in the paper "MARCUS: An agentic, multimodal vision-language model for cardiac diagnosis and management" (O'Sullivan et al., 2026).

The model takes as input a hospital-style 12-lead ECG PNG (4-row × 3-lead layout, 25 mm/s paper speed, 10 mm/mV gain) and produces free-text clinical interpretations or multiple-choice answers covering rhythm, rate, axis, conduction abnormalities, ST-T changes, and other ECG findings.

---

## Architecture

```
12-lead ECG (12 × N float array, mV)
        │
        ▼  ECG renderer (25 mm/s, 10 mm/mV, 10 s, 4×3 grid)
Hospital-style PNG (4 rows × 3 leads, 224 × 224 px per patch)
        │
        ▼
SigLIP Vision Encoder
        │
        ▼
2-Layer MLP Projection Head
        │
        ▼
Qwen2 Language Model (3B, decoder-only)
        │
        ▼
Free-text clinical interpretation
```

### Key Architectural Details

- **Vision encoder:** SigLIP (Sigmoid Loss for Language-Image Pre-training), patch-based
- **Projection:** 2-layer MLP mapping visual tokens to the LLM embedding dimension
- **Language model:** Qwen2 3B decoder, instruction-tuned
- **Image resolution:** 224 × 224 px per lead patch; full grid ~900 × 600 px

---

## Training Data

| Property                | Value                                        |
|-------------------------|----------------------------------------------|
| Dataset size            | 249,785 12-lead ECGs                         |
| Source institution      | Stanford University Medical Center           |
| Paired annotations      | Physician-written free-text interpretation reports |
| Preprocessing           | XML/numeric → hospital-style PNG             |
| Split                   | Training / validation / internal test        |

### Data Collection

ECGs were collected from routine clinical practice at Stanford University Medical Center. Each ECG is paired with a physician-authored narrative interpretation report generated at the time of clinical reading. ECGs were recorded using standard 12-lead acquisition protocols (10 seconds, 500 Hz sampling rate).

### Preprocessing Pipeline

1. Raw ECG signal (`.npy` shape `(12, N)` or PhilipsXML) is loaded.
2. Signals are band-pass filtered (0.5–40 Hz) and baseline-corrected.
3. A hospital-style PNG is rendered: 4 rows of 3 leads, 25 mm/s paper speed, 10 mm/mV gain, standard lead ordering (I, II, III / aVR, aVL, aVF / V1–V6).
4. Lead patches are 224 × 224 px; grid lines, lead labels, and scale bars are drawn.

---

## Training Procedure

| Stage | Method       | Trainable Components         | Data                            | Objective                         |
|-------|--------------|------------------------------|---------------------------------|-----------------------------------|
| 1     | Pretraining  | Vision encoder + MLP head    | (ECG PNG, physician report)     | Next-token prediction on reports  |
| 2     | SFT          | Full model                   | (ECG PNG, question, answer) MCQ+VQA | Cross-entropy on answer tokens |

No GRPO stage was applied to the ECG expert (Stage 3 GRPO was applied to the Echo and CMR experts).

### Training Configuration

- Framework: [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- Inference/serving: `llamafactory-cli api` with `examples/inference/qwen2_5vl.yaml`
- Default checkpoint path: `saves/Qwen2.5-VL-3B-Instruct/full/ecg_sft`

---

## Evaluation

### MCQ Accuracy

| Evaluation Set    | Accuracy |
|-------------------|----------|
| Stanford internal | 87%      |
| UCSF external     | 91%      |

MCQ accuracy is computed on 4-choice questions derived from the MARCUS-Benchmark dataset. Excluded items (where the judge cannot determine correctness) are removed from the denominator.

### VQA Likert Score

| Evaluation Set    | Mean Likert (1–5) |
|-------------------|--------------------|
| Stanford internal | 3.65               |

VQA Likert scores are assigned by a GPT-4-class judge evaluating clinical accuracy, completeness, and relevance on a 1–5 scale.

### Comparison with Frontier Models

| Model                     | Stanford MCQ | UCSF MCQ |
|---------------------------|--------------|----------|
| MARCUS-ECG                | **87%**      | **91%**  |
| GPT-5 Thinking            | ~48%         | —        |
| Gemini 2.5 Pro Deep Think | ~51%         | —        |

### Mirage Rate

MARCUS-ECG achieves a **0% mirage rate** when deployed within the MARCUS agentic orchestrator (counterfactual verification protocol). When used standalone without mirage probing, the model's hallucination rate is not separately characterized.

---

## Intended Use

### Primary Use Cases

- **Research:** Benchmarking ECG interpretation capabilities of vision-language models.
- **Clinical decision support:** Providing a second opinion or structured interpretation to assist clinically trained cardiologists and electrophysiologists.
- **Medical education:** Generating educational interpretations for ECG training.

### Out-of-Scope Uses

- Autonomous clinical diagnosis without physician oversight.
- Use in emergency triage or time-critical clinical workflows without validation.
- Use outside the intended input format (e.g., non-standard ECG lead configurations).
- Pediatric ECG interpretation (training data is predominantly adult).

---

## Limitations

### Data Limitations

- **Single-center training:** All training data originates from Stanford University Medical Center. The model may underperform on ECGs from institutions with different acquisition hardware, patient populations, or reporting styles.
- **Curated evaluation:** The MARCUS-Benchmark is derived from clinical records and has undergone quality filtering (B-Clean protocol). Performance on unfiltered real-world ECGs may differ.
- **Rare diagnoses:** Uncommon ECG patterns (e.g., rare channelopathies, device-related artifacts) are underrepresented in the training data.
- **Pediatric ECGs:** Training data is predominantly adult; pediatric age-specific normal ranges are not validated.

### Model Limitations

- **No temporal context:** The model processes a single 10-second ECG. Serial comparison and trend analysis are not supported.
- **No demographic conditioning:** Patient age, sex, and medications — which affect ECG interpretation — are not currently provided as model inputs.
- **Free-text variability:** VQA outputs are not standardized; post-processing may be needed for downstream applications requiring structured output.

---

## Bias and Ethics Considerations

### Potential Biases

- **Geographic and demographic bias:** Training data from a single U.S. academic medical center may not represent global ECG patterns, body habitus effects, or population-specific prevalence of cardiac conditions.
- **Reporting style bias:** The model learns to mimic the reporting style of Stanford physicians. Findings emphasized or de-emphasized in that reporting culture may be over- or under-represented in model outputs.
- **Confirmation bias in labels:** Physician reports used as ground truth reflect the clinical context at the time of interpretation and may contain errors or incomplete interpretations.

### Clinical Safety

- MARCUS-ECG is **not an FDA-cleared medical device** and should not be used as the sole basis for clinical decisions.
- All outputs should be reviewed by a qualified clinician before clinical action is taken.
- The model may produce plausible-sounding but incorrect interpretations. Users should apply appropriate skepticism, particularly for rare or ambiguous findings.

### Privacy

- No patient data is stored or transmitted by the model weights or the MARCUS inference stack.
- Input ECG images are processed locally; no data is sent to external servers unless explicitly configured (e.g., when using the GPT judge for evaluation).

---

## How to Use

### CLI

```bash
# Start the ECG expert API + web UI
video-chat-ecg
# UI available at http://localhost:8775
```

### Python API

```python
import httpx, base64, json

# Load and preprocess ECG
import numpy as np
from video_chat_ui.preprocessing.ecg import render_ecg_png

ecg = np.load("patient.npy")            # shape (12, N), float32, mV
render_ecg_png(ecg, "ecg_grid.png")     # hospital-style PNG

# Upload and query
client = httpx.Client(base_url="http://localhost:8020")

with open("ecg_grid.png", "rb") as f:
    resp = client.post("/upload", files={"file": f})
media_id = resp.json()["id"]

answer = client.post("/chat", json={
    "message": "What is the cardiac rhythm and are there any conduction abnormalities?",
    "video_id": media_id,
    "media_kind": "image",
}).json()["reply"]

print(answer)
```

### Direct via LLaMA-Factory API

```python
import httpx, base64

with open("ecg_grid.png", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

response = httpx.post(
    "http://localhost:8020/v1/chat/completions",
    json={
        "model": "ecg",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text",
                 "text": "Interpret this 12-lead ECG."}
            ]
        }]
    }
)
print(response.json()["choices"][0]["message"]["content"])
```

---

## Citation

```bibtex
@article{osullivan2026marcus,
  title   = {{MARCUS}: An agentic, multimodal vision-language model for cardiac diagnosis and management},
  author  = {O'Sullivan, Jack W and Asadi, Mohammad and Elbe, Lennart and others},
  journal = {arXiv preprint},
  year    = {2026},
  url     = {https://arxiv.org/abs/XXXX.XXXXX}
}
```

---

*This model card follows the [Model Card](https://arxiv.org/abs/1810.03993) framework (Mitchell et al., 2019).*
