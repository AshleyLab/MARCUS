# MARCUS Architecture

## Overview

MARCUS (Multimodal Autonomous Reasoning and Chat for Ultrasound and Signals) is an agentic medical AI system composed of three specialized expert models and a coordinating orchestrator. Each expert is a 3-billion-parameter vision-language model fine-tuned from Qwen 2.5-VL-3B-Instruct on one cardiac imaging modality. The orchestrator is a stateless agentic layer that interfaces with all three experts via an OpenAI-compatible chat completion API, decomposes clinical queries, aggregates expert responses, and applies counterfactual probing to detect and suppress mirage reasoning.

```
                          ┌────────────────────────────────────────┐
                          │          Clinical Input                │
                          │  (ECG PNG / Echo video / CMR video /   │
                          │   free-text query / combined)          │
                          └──────────────────┬─────────────────────┘
                                             │
                          ┌──────────────────▼─────────────────────┐
                          │          Agentic Orchestrator           │
                          │                                         │
                          │  1. Query Decomposition                 │
                          │     Break clinical question into        │
                          │     modality-specific sub-queries       │
                          │                                         │
                          │  2. Expert Routing                      │
                          │     Route each sub-query + media to     │
                          │     appropriate expert via REST API     │
                          │                                         │
                          │  3. Mirage Probing                      │
                          │     Counterfactual (image-absent)       │
                          │     consistency check per expert        │
                          │                                         │
                          │  4. Confidence Scoring                  │
                          │     Per-modality confidence from        │
                          │     rephrase agreement + image delta    │
                          │                                         │
                          │  5. Answer Aggregation                  │
                          │     Confidence-weighted synthesis of    │
                          │     all expert responses                │
                          └───┬──────────────┬──────────────┬──────┘
                              │              │              │
             ┌────────────────▼──┐  ┌────────▼───────┐  ┌──▼────────────────┐
             │    ECG Expert     │  │  Echo Expert   │  │    CMR Expert     │
             │                   │  │                │  │                   │
             │  SigLIP Encoder   │  │  ViT Encoder   │  │  ViT Encoder      │
             │  2-layer MLP proj │  │  Temporal agg  │  │  Metadata routing │
             │  Qwen2 LM         │  │  Cross-view    │  │  Qwen2 LM         │
             │                   │  │  Qwen2 LM      │  │                   │
             │  Port: 8020       │  │  Port: 8010    │  │  Port: 8000       │
             └───────────────────┘  └────────────────┘  └───────────────────┘
```

---

## Expert Models

All three expert models share the same base: **Qwen/Qwen2.5-VL-3B-Instruct**. They differ in their visual preprocessing pipeline, training data, and fine-tuning recipe.

---

### ECG Expert Model

**Checkpoint:** `saves/Qwen2.5-VL-3B-Instruct/full/ecg_sft`
**HuggingFace:** `stanford-cardiac-ai/MARCUS-ECG`
**Training:** SFT only (Stages 1 and 2; no GRPO)

#### Architecture

```
12-lead ECG waveform (12 × N float array, mV)
         │
         ▼
  ECG Renderer (25 mm/s, 10 mm/mV, 10 s)
  4-row × 3-lead hospital-style PNG
  224 × 224 px per lead patch
         │
         ▼
  SigLIP Vision Encoder
  (patch-based, ViT-style, pretrained on biomedical images)
         │
         ▼
  2-Layer MLP Projection Head
  (maps vision tokens to LLM embedding dimension)
         │
         ▼
  Qwen2 Language Model
  (3B parameter decoder; generates free-text clinical interpretation)
```

#### Input Format

- **File format:** `.npy` (shape `(12, N)`, float32, units mV) or XML (PhilipsXML / Muse)
- **Rendered format:** Hospital-style 12-lead ECG PNG
- **Paper speed:** 25 mm/s (standard clinical convention)
- **Gain:** 10 mm/mV
- **Duration:** 10 seconds
- **Layout:** 4 rows × 3 leads (standard clinical layout)
- **Patch size:** 224 × 224 px per lead

#### Training Data

- 249,785 12-lead ECGs with paired physician interpretation reports
- Source: Stanford University Medical Center
- Preprocessing: XML/numeric → hospital-style PNG via `video_chat_ui.preprocessing.ecg`

#### Training Recipe

| Stage | Method | Details                                                        |
|-------|--------|----------------------------------------------------------------|
| 1     | Encoder pretraining | Freeze LLM; train vision encoder + MLP on (ECG, report) pairs |
| 2     | SFT    | Fine-tune full model on MCQ + VQA instruction pairs           |

#### Performance

| Metric           | Stanford | UCSF |
|------------------|----------|------|
| MCQ Accuracy     | 87%      | 91%  |
| VQA Likert (1–5) | 3.65     | —    |

---

### Echocardiography Expert Model

**Checkpoint:** `saves/Qwen2.5-VL-3B-Instruct/full/echo_grpo`
**HuggingFace:** `stanford-cardiac-ai/MARCUS-Echo`
**Training:** SFT + GRPO (all 3 stages)

#### Architecture

```
Multi-view Echo DICOM study (.tgz)
         │
         ▼
  DICOM Extractor + View Selector
  (attention-based, no manual annotation)
         │
         ▼
  Multi-view Video Grid Assembly
  (key views assembled into grid MP4)
         │
         ▼
  Visual Encoder (ViT, 16 × 16 patches)
  per-frame spatial encoding
         │
         ▼
  Temporal Aggregation Module
  (pools frame-level features across time)
         │
         ▼
  Cross-View Fusion Module
  (combines features across views via cross-attention)
         │
         ▼
  Adapter Layer (Cross-Attention)
  vision embeddings cross-attend to text tokens
  residual connections between ViT layers and LLM blocks
         │
         ▼
  Qwen2 Language Model
```

#### Input Format

- **File format:** Multi-view DICOM `.tgz` (standard clinical export)
- **Rendered format:** Multi-view grid video (MP4 or AVI)
- **View selection:** Attention-based automatic selection; no manual labeling required
- **Patch decomposition:** 16 × 16 spatial patches per frame

#### Training Data

- 1,266,144 echocardiography images from 10,823 studies
- Source: Stanford University Medical Center
- Views include: parasternal long axis, parasternal short axis, apical 4-chamber, apical 2-chamber, subcostal, and others

#### Training Recipe

| Stage | Method | Details                                                                  |
|-------|--------|--------------------------------------------------------------------------|
| 1     | Encoder pretraining | Freeze LLM; train visual encoder on (echo frame, report) pairs  |
| 2     | SFT    | Full fine-tuning on MCQ + VQA pairs from physician reports               |
| 3     | GRPO   | Group Relative Policy Optimization with MCQ binary correctness reward    |

#### Performance

| Metric           | Stanford | UCSF |
|------------------|----------|------|
| MCQ Accuracy     | 67%      | 86%  |
| VQA Likert (1–5) | 2.41     | —    |

---

### CMR Expert Model

**Checkpoint:** `saves/Qwen2.5-VL-3B-Instruct/full/cmr_grpo`
**HuggingFace:** `stanford-cardiac-ai/MARCUS-CMR`
**Training:** SFT + GRPO (all 3 stages)

#### Architecture

```
Multi-sequence CMR DICOM study (.tgz)
         │
         ▼
  DICOM Extractor + Metadata Parser
  (reads SeriesDescription, SequenceName, etc.)
         │
         ▼
  Metadata-Driven Sequence Router
  (selects cine / LGE / T2 / T1 / other based on query)
         │
         ▼
  Multi-Slice Grid Assembly
  (selected sequences assembled into grid MP4 via OpenCV)
         │
         ▼
  Visual Encoder (ViT, 16 × 16 patches)
         │
         ▼
  Adapter Layer (Cross-Attention)
  vision embeddings → cross-attention → concat with text tokens
  residual connections between ViT layers and LLM blocks
         │
         ▼
  Qwen2 Language Model
```

#### Input Format

- **File format:** Multi-sequence, multi-slice DICOM `.tgz`
- **Rendered format:** Multi-slice grid video (MP4, OpenCV-only; no FFmpeg required)
- **Sequences supported:** Cine (SSFP), LGE, T2-weighted, T1-mapping, T1-weighted, phase-contrast
- **Metadata routing:** SeriesDescription and SequenceName DICOM tags drive sequence selection

#### Training Data

- 12,191,751 CMR images from 9,473 studies
- Source: Stanford University Medical Center
- Sequences: cine short-axis, cine long-axis, LGE, T2 STIR, T1 mapping, and others

#### Training Recipe

Identical to Echo: Stage 1 encoder pretraining → Stage 2 SFT → Stage 3 GRPO.

#### Performance

| Metric           | Stanford | UCSF |
|------------------|----------|------|
| MCQ Accuracy     | 88%      | 85%  |
| VQA Likert (1–5) | 2.91     | —    |

---

### Adapter Layer (Shared Design)

Both the Echo and CMR experts use a cross-attention adapter between the visual encoder and the language model:

```
Vision Encoder output (sequence of patch embeddings)
         │
         ▼
  Cross-Attention Layer
  ┌─────────────────────────────────────────┐
  │  Query: text token embeddings           │
  │  Key/Value: vision patch embeddings     │
  │  Output: vision-enriched text tokens    │
  └─────────────────────────────────────────┘
         │
         ▼  (+ residual from ViT layer output)
  Concatenated with text token sequence
         │
         ▼
  Qwen2 Transformer Blocks
```

Residual connections carry visual information from each ViT block into the corresponding LLM block, enabling layer-wise visual grounding.

---

## Agentic Orchestrator

The orchestrator is a Python process that coordinates the three experts via HTTP. It does not have its own learned weights; its behavior is governed by prompt engineering and deterministic logic.

### Query Decomposition

When a user submits a multimodal clinical query (e.g., "Summarize this patient's cardiac workup"), the orchestrator uses a chain-of-thought decomposition prompt to break it into modality-specific sub-queries:

```
Input:  "What are the key findings from this patient's ECG, echo, and CMR?"
Output:
  ecg_query:  "Identify the rhythm, rate, axis, and any interval abnormalities."
  echo_query: "Report ventricular function, wall motion, and valve status."
  cmr_query:  "Describe myocardial morphology, LGE pattern, and any fibrosis."
```

### Expert Routing

Each sub-query and its associated media file are routed to the appropriate expert via the OpenAI-compatible `/v1/chat/completions` endpoint:

```python
# Pseudocode
ecg_response  = post_to_api(ecg_url,  ecg_query,  ecg_image)
echo_response = post_to_api(echo_url, echo_query, echo_video)
cmr_response  = post_to_api(cmr_url,  cmr_query,  cmr_video)
```

### Confidence Scoring

Per-modality confidence is a scalar in [0, 1] computed from two components:

1. **Rephrase agreement** (α): Three semantically equivalent rephrasings of each sub-query are sent to the expert. Agreement across responses is measured by token-level n-gram overlap (ROUGE-L). High agreement → high confidence.

2. **Image-delta score** (β): The same query is sent *without* the image. The cosine distance between image-present and image-absent response embeddings is computed. Large distance → the model is actually using the image → high confidence.

```
confidence = α * rephrase_agreement + (1 - α) * image_delta_score
```

If `confidence < threshold`, the answer is flagged as low-confidence and a disclaimer is prepended.

### Mirage Probing

See [Mirage Resistance](#mirage-resistance) below.

### Answer Aggregation

Final answer = confidence-weighted synthesis of expert responses, composed via a final LLM call with a synthesis prompt that combines the sub-answers and their confidence levels into a coherent clinical summary.

---

## Training Pipeline

### Stage 1: Vision Encoder Pretraining

```
┌────────────────────────────────────────────────────────┐
│  Data: (image/video, physician report) pairs           │
│                                                        │
│  Trainable:  Vision encoder + MLP projection head      │
│  Frozen:     Qwen2 language model                      │
│                                                        │
│  Objective:  Next-token prediction on report text      │
│              conditioned on visual tokens              │
│                                                        │
│  Purpose:    Align visual representation space with    │
│              medical report vocabulary and concepts    │
└────────────────────────────────────────────────────────┘
```

### Stage 2: Supervised Fine-Tuning (SFT)

```
┌────────────────────────────────────────────────────────┐
│  Data: (image/video, question, answer) instruction     │
│        triples — MCQ and VQA formats                   │
│                                                        │
│  Trainable:  Full model (encoder + projection + LLM)   │
│                                                        │
│  Objective:  Cross-entropy loss on answer tokens       │
│                                                        │
│  Purpose:    Task-specific instruction following;      │
│              adapt to MCQ/VQA answer format            │
└────────────────────────────────────────────────────────┘
```

### Stage 3: GRPO Alignment

```
┌────────────────────────────────────────────────────────┐
│  Data: MCQ questions with ground-truth labels          │
│                                                        │
│  Algorithm: Group Relative Policy Optimization (GRPO)  │
│                                                        │
│  Reward:    +1 if MCQ answer matches ground truth      │
│              0 otherwise                               │
│                                                        │
│  Purpose:   Improve calibration, reduce overconfidence │
│             on unseen questions, suppress mirage rate  │
└────────────────────────────────────────────────────────┘
```

GRPO samples multiple completions per question (the "group"), estimates a baseline from the group mean reward, and updates the policy with a clipped ratio objective — analogous to PPO but without a learned value function.

---

## Mirage Resistance

*Mirage reasoning* is the phenomenon where a VLM produces confidently stated, clinically plausible findings that are not actually supported by the input image. MARCUS achieves a **0% mirage rate** on the benchmark through a three-step protocol.

### Step 1: Rephrase Consistency Check

Each clinical query is rephrased three times using a paraphrase prompt. All four variants (original + 3 rephrasings) are sent to the expert with the image. A mirage-prone model will give inconsistent answers across rephrasings even when the image is the same — because it is responding to spurious text cues rather than visual content.

Consistency score = mean pairwise ROUGE-L over the four responses.

### Step 2: Image-Absent Counterfactual Probe

The original query is sent to the expert *without* attaching the image. A model that produces a nearly identical answer with and without the image is not grounding its response in the visual input — it is hallucinating (mirage).

```
image_delta = cosine_distance(
    embed(response_with_image),
    embed(response_without_image)
)
```

High `image_delta` means the model's answer meaningfully changes when the image is removed — a sign of genuine visual grounding.

### Step 3: Confidence Score and Gating

```python
confidence = alpha * consistency_score + (1 - alpha) * image_delta
if confidence < MIRAGE_THRESHOLD:
    response = prepend_disclaimer(response)
    # "Note: confidence in this response is low. ..."
```

`alpha` and `MIRAGE_THRESHOLD` are set empirically on a held-out calibration set. The combination of rephrase consistency and image-delta eliminates mirages while preserving high-confidence genuine responses.

---

## Port Reference

| Expert | API Port | UI Port | Checkpoint          |
|--------|----------|---------|---------------------|
| CMR    | 8000     | 8765    | `cmr_grpo`          |
| Echo   | 8010     | 8770    | `echo_grpo`         |
| ECG    | 8020     | 8775    | `ecg_sft`           |

All APIs expose the OpenAI-compatible `/v1/chat/completions` and `/v1/models` endpoints served by LLaMA-Factory.
