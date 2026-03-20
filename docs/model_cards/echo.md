# Model Card: MARCUS-Echo

## Model Overview

| Field             | Value                                                                   |
|-------------------|-------------------------------------------------------------------------|
| Model ID          | `stanford-cardiac-ai/MARCUS-Echo`                                        |
| Base model        | `Qwen/Qwen2.5-VL-3B-Instruct`                                           |
| Parameters        | 3 billion                                                               |
| Modality          | Echocardiography (transthoracic and transesophageal)                    |
| Input             | Multi-view echocardiography DICOM video (assembled grid MP4)            |
| Output            | Free-text clinical interpretation or MCQ answer                         |
| Training method   | Stage 1 encoder pretraining + Stage 2 SFT + Stage 3 GRPO               |
| License           | MIT (model weights: Qwen License)                                       |
| HuggingFace       | [stanford-cardiac-ai/MARCUS-Echo](https://huggingface.co/stanford-cardiac-ai/MARCUS-Echo) |

---

## Description

MARCUS-Echo is a 3-billion-parameter vision-language model specialized for echocardiography interpretation. It is one of three expert components of the MARCUS cardiac AI system, described in the paper "MARCUS: An agentic, multimodal vision-language model for cardiac diagnosis and management" (O'Sullivan et al., 2026).

The model accepts multi-view echocardiography video input (preprocessed from clinical DICOM studies) and generates clinical interpretations covering left and right ventricular function, wall motion abnormalities, valvular disease, pericardial effusion, and other standard echocardiographic findings.

---

## Architecture

```
Multi-view Echo DICOM study (.tgz)
         │
         ▼  DICOM extractor + attention-based view selector
Key echocardiographic views (no manual annotation required)
         │
         ▼  Grid assembly (FFmpeg MP4 / OpenCV AVI)
Multi-view grid video
         │
         ▼
ViT Visual Encoder (16 × 16 patch decomposition, per-frame)
         │
         ▼
Temporal Aggregation Module
(pools frame-level features across cardiac cycles)
         │
         ▼
Cross-View Fusion Module
(cross-attention across simultaneous views)
         │
         ▼
Adapter Layer (Cross-Attention)
vision embeddings × text token queries
+ residual connections between ViT layers and LLM blocks
         │
         ▼
Qwen2 Language Model (3B, decoder-only)
         │
         ▼
Free-text clinical interpretation
```

### Key Architectural Details

- **Visual encoder:** ViT with 16 × 16 spatial patch decomposition
- **Temporal aggregation:** Learnable pooling over frames within each cardiac cycle
- **Cross-view fusion:** Cross-attention module combining features from multiple simultaneously acquired views
- **Adapter:** Cross-attention adapter with residual connections between ViT and LLM blocks
- **Language model:** Qwen2 3B decoder

---

## Training Data

| Property                | Value                                                  |
|-------------------------|--------------------------------------------------------|
| Dataset size            | 1,266,144 echocardiography images from 10,823 studies  |
| Source institution      | Stanford University Medical Center                     |
| Paired annotations      | Physician-written structured echocardiography reports  |
| Views included          | Parasternal long/short axis, apical 2/4/5-chamber, subcostal, suprasternal |
| Preprocessing           | DICOM `.tgz` → view selection → grid video assembly    |

### Data Collection

Echocardiography studies were collected from routine clinical practice at Stanford University Medical Center. Studies include transthoracic echocardiography (TTE) from a range of clinical indications. Each study is paired with a physician-authored structured report generated at the time of clinical interpretation.

### Preprocessing Pipeline

1. DICOM `.tgz` is extracted and per-series DICOM files are parsed.
2. An attention-based view selector identifies the most diagnostically informative views without manual annotation.
3. Selected view clips are assembled into a multi-view grid video.
4. MP4 encoding is applied if FFmpeg is available; AVI fallback is used otherwise.

---

## Training Procedure

| Stage | Method       | Trainable Components          | Data                                | Objective                         |
|-------|--------------|-------------------------------|-------------------------------------|-----------------------------------|
| 1     | Pretraining  | Vision encoder + adapter      | (echo frame, report) pairs          | Next-token prediction on reports  |
| 2     | SFT          | Full model                    | (echo video, question, answer) MCQ+VQA | Cross-entropy on answer tokens |
| 3     | GRPO         | Full model                    | MCQ with binary correctness reward  | Policy optimization               |

### Training Configuration

- Stages 1 & 2: [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- Stage 3: [verl](https://github.com/volcengine/verl) with sglang rollout backend
- Default checkpoint path: `saves/Qwen2.5-VL-3B-Instruct/full/echo_grpo`

---

## Evaluation

### MCQ Accuracy

| Evaluation Set    | Accuracy |
|-------------------|----------|
| Stanford internal | 67%      |
| UCSF external     | 86%      |

### VQA Likert Score

| Evaluation Set    | Mean Likert (1–5) |
|-------------------|--------------------|
| Stanford internal | 2.41               |

VQA Likert scores are assigned by a GPT-4-class judge evaluating clinical accuracy, completeness, and relevance on a 1–5 scale.

### Comparison with Frontier Models

| Model                     | Stanford MCQ | UCSF MCQ |
|---------------------------|--------------|----------|
| MARCUS-Echo               | **67%**      | **86%**  |
| GPT-5 Thinking            | ~34%         | —        |
| Gemini 2.5 Pro Deep Think | ~42%         | —        |

### Mirage Rate

MARCUS-Echo achieves a **0% mirage rate** when deployed within the MARCUS agentic orchestrator (counterfactual verification protocol).

---

## Intended Use

### Primary Use Cases

- **Research:** Benchmarking echo interpretation capabilities of vision-language models.
- **Clinical decision support:** Providing structured interpretation assistance to echocardiographers and cardiologists.
- **Automated reporting:** Generating draft reports for physician review.
- **Quality assurance:** Flagging potentially missed findings for secondary review.

### Out-of-Scope Uses

- Autonomous clinical diagnosis without physician oversight.
- Stress echocardiography or intraoperative TEE interpretation (not represented in training data).
- Interpretation of contrast echocardiography or strain imaging.
- Use in pediatric echocardiography (training data is predominantly adult).

---

## Limitations

### Data Limitations

- **Single-center training:** All training data originates from Stanford University Medical Center.
- **View coverage:** Studies missing standard views (poor acoustic windows) may produce degraded output.
- **Quantitative measurements:** The model generates qualitative interpretations; numerical measurements (EF%, valve gradients) should be validated against dedicated quantification software.
- **Rare pathologies:** Complex congenital heart disease, rare cardiomyopathies, and unusual artifact patterns are underrepresented.

### Model Limitations

- **Video duration sensitivity:** Very short clips (< 1 cardiac cycle) or very long clips may degrade performance.
- **View selection errors:** The attention-based view selector may occasionally select suboptimal views in studies with degraded image quality.
- **Temporal resolution:** The model processes a fixed number of frames per clip; very high frame-rate acquisitions are downsampled.
- **No integration with measurements:** The model does not have access to DICOM-derived quantitative measurements (EF, LA volume, etc.) during inference.

---

## Bias and Ethics Considerations

### Potential Biases

- **Demographic bias:** Training population reflects the demographics of patients seen at Stanford. Performance on populations with different disease prevalence (e.g., higher rates of rheumatic heart disease) is not validated.
- **Equipment bias:** Training data reflects image quality and acquisition parameters of equipment at Stanford. Studies from different vendors or older equipment may show reduced performance.
- **Interpretive style:** Model outputs reflect the interpretive conventions of Stanford echocardiographers.

### Clinical Safety

- MARCUS-Echo is **not an FDA-cleared medical device**.
- Echo interpretation requires integration of clinical context not available to the model. All outputs should be reviewed by a board-certified echocardiographer.
- Do not use for primary screening of valvular heart disease or other conditions requiring high sensitivity without physician review.

### Privacy

- No patient data is stored or transmitted by the model weights or inference stack.
- DICOM input is processed locally; metadata (patient name, ID, date) is not used as model input.

---

## How to Use

### CLI

```bash
# Start the Echo expert API + web UI
video-chat-echo
# UI available at http://localhost:8770
```

### Python API

```python
import httpx
from video_chat_ui.preprocessing.echo import process_echo_study

# Preprocess DICOM study
process_echo_study("echo_study.tgz", "echo_grid.mp4", workdir="/tmp")

# Upload and query
client = httpx.Client(base_url="http://localhost:8010")

with open("echo_grid.mp4", "rb") as f:
    resp = client.post("/upload", files={"file": f})
media_id = resp.json()["id"]

answer = client.post("/chat", json={
    "message": "Describe left ventricular function and any wall motion abnormalities.",
    "video_id": media_id,
    "media_kind": "video",
}).json()["reply"]

print(answer)
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
