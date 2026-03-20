# Model Card: MARCUS-CMR

## Model Overview

| Field             | Value                                                                  |
|-------------------|------------------------------------------------------------------------|
| Model ID          | `stanford-cardiac-ai/MARCUS-CMR`                                        |
| Base model        | `Qwen/Qwen2.5-VL-3B-Instruct`                                          |
| Parameters        | 3 billion                                                              |
| Modality          | Cardiac Magnetic Resonance Imaging (CMR)                               |
| Input             | Multi-sequence, multi-slice CMR DICOM study (assembled grid MP4)       |
| Output            | Free-text clinical interpretation or MCQ answer                        |
| Training method   | Stage 1 encoder pretraining + Stage 2 SFT + Stage 3 GRPO              |
| License           | MIT (model weights: Qwen License)                                      |
| HuggingFace       | [stanford-cardiac-ai/MARCUS-CMR](https://huggingface.co/stanford-cardiac-ai/MARCUS-CMR) |

---

## Description

MARCUS-CMR is a 3-billion-parameter vision-language model specialized for cardiac MRI interpretation. It is one of three expert components of the MARCUS cardiac AI system, described in the paper "MARCUS: An agentic, multimodal vision-language model for cardiac diagnosis and management" (O'Sullivan et al., 2026).

The model accepts multi-sequence CMR studies (preprocessed from clinical DICOM data) and generates clinical interpretations covering ventricular morphology and function, myocardial fibrosis and scar (LGE), edema (T2), and other standard CMR findings across cine, LGE, T2, and T1 sequences.

---

## Architecture

```
Multi-sequence CMR DICOM study (.tgz)
         │
         ▼  DICOM extractor + DICOM metadata parser
         │  (SeriesDescription, SequenceName, etc.)
         │
         ▼  Metadata-Driven Sequence Router
         │  selects cine / LGE / T2 / T1 / other
         │  based on clinical query context
         │
         ▼  Multi-slice grid assembly (OpenCV; no FFmpeg required)
Multi-sequence grid video
         │
         ▼
ViT Visual Encoder (16 × 16 patch decomposition, per-frame)
         │
         ▼
Adapter Layer (Cross-Attention)
vision embeddings → cross-attention → concat with text tokens
+ residual connections between ViT layers and LLM blocks
         │
         ▼
Qwen2 Language Model (3B, decoder-only)
         │
         ▼
Free-text clinical interpretation
```

### Key Architectural Details

- **Visual encoder:** ViT with 16 × 16 spatial patch decomposition per frame
- **Sequence routing:** DICOM metadata tags (`SeriesDescription`, `SequenceName`, `ImageType`) drive sequence selection before the model sees any pixel
- **Adapter:** Cross-attention adapter with residual connections between ViT and LLM blocks
- **Video assembly:** OpenCV-based grid assembly (no FFmpeg dependency); CMR sequences are spatially co-registered per slice before display
- **Language model:** Qwen2 3B decoder

---

## Training Data

| Property                | Value                                                           |
|-------------------------|-----------------------------------------------------------------|
| Dataset size            | 12,191,751 CMR images from 9,473 studies                        |
| Source institution      | Stanford University Medical Center                              |
| Paired annotations      | Physician-written CMR reports (structured + narrative)          |
| Sequences included      | Cine SSFP (short-axis, long-axis), LGE, T2-STIR, T1-mapping, T1-weighted, phase-contrast |
| Preprocessing           | DICOM `.tgz` → metadata routing → per-series grid → grid video  |

### Data Collection

CMR studies were collected from routine clinical practice at Stanford University Medical Center. The dataset spans a broad range of clinical indications including ischemic cardiomyopathy, non-ischemic cardiomyopathy, myocarditis, hypertrophic cardiomyopathy, cardiac sarcoidosis, amyloidosis, and congenital heart disease. Each study is paired with a physician-authored report generated at the time of clinical reading.

### Preprocessing Pipeline

1. DICOM `.tgz` is extracted; per-series DICOM stacks are parsed.
2. DICOM metadata tags (`SeriesDescription`, `SequenceName`, `SliceLocation`, `InstanceNumber`) are read to identify sequence type and spatial ordering.
3. A metadata-driven router selects appropriate sequences based on the clinical query (e.g., LGE sequences for scar queries, T2 for edema queries).
4. Selected sequences are assembled into a multi-slice grid video using OpenCV. Slices are sorted by spatial location and stacked into a spatial-temporal grid.
5. Output is an MP4 (or AVI fallback) grid video representing the full spatial coverage of the study.

---

## Training Procedure

| Stage | Method       | Trainable Components          | Data                                | Objective                         |
|-------|--------------|-------------------------------|-------------------------------------|-----------------------------------|
| 1     | Pretraining  | Vision encoder + adapter      | (CMR frame, report) pairs           | Next-token prediction on reports  |
| 2     | SFT          | Full model                    | (CMR video, question, answer) MCQ+VQA | Cross-entropy on answer tokens |
| 3     | GRPO         | Full model                    | MCQ with binary correctness reward  | Policy optimization               |

### Training Configuration

- Stages 1 & 2: [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- Stage 3: [verl](https://github.com/volcengine/verl) with sglang rollout backend
- Default checkpoint path: `saves/Qwen2.5-VL-3B-Instruct/full/cmr_grpo`

---

## Evaluation

### MCQ Accuracy

| Evaluation Set    | Accuracy |
|-------------------|----------|
| Stanford internal | 88%      |
| UCSF external     | 85%      |

MCQ accuracy is computed on 4-choice questions derived from the MARCUS-Benchmark dataset. Excluded items are removed from the denominator.

### VQA Likert Score

| Evaluation Set    | Mean Likert (1–5) |
|-------------------|--------------------|
| Stanford internal | 2.91               |

### Comparison with Frontier Models

| Model                     | Stanford MCQ | UCSF MCQ |
|---------------------------|--------------|----------|
| MARCUS-CMR                | **88%**      | **85%**  |
| GPT-5 Thinking            | ~58%         | —        |
| Gemini 2.5 Pro Deep Think | ~41%         | —        |

### Mirage Rate

MARCUS-CMR achieves a **0% mirage rate** when deployed within the MARCUS agentic orchestrator (counterfactual verification protocol).

---

## Intended Use

### Primary Use Cases

- **Research:** Benchmarking CMR interpretation capabilities of vision-language models.
- **Clinical decision support:** Providing structured interpretation assistance to cardiac radiologists and cardiologists.
- **Automated pre-read:** Generating a draft interpretation for physician review, particularly in high-volume settings.
- **Education:** Teaching CMR interpretation patterns to trainees.

### Out-of-Scope Uses

- Autonomous clinical diagnosis without physician oversight.
- Real-time intraoperative CMR guidance.
- Quantitative measurement extraction (EF, volumes, strain) without validation against dedicated segmentation software.
- Pediatric CMR interpretation (training data is predominantly adult with adult normal ranges).

---

## Limitations

### Data Limitations

- **Single-center training:** All training data originates from Stanford University Medical Center. Protocol variation across institutions (field strength, vendor, sequence parameters) may affect performance.
- **Protocol variability:** The model is trained on standard clinical CMR protocols. Non-standard sequences, research protocols, or non-cardiac MRI series may degrade output.
- **Rare diagnoses:** Unusual cardiomyopathies and rare genetic syndromes are underrepresented.
- **Gadolinium dependence for LGE:** LGE interpretation is contingent on adequate contrast administration; the model cannot flag inadequate contrast timing from the image alone.

### Model Limitations

- **No quantitative output:** The model generates qualitative assessments. Quantitative measurements (LVEF, LV mass, LGE burden percentage) require dedicated software.
- **Metadata routing dependency:** Incorrect or non-standard DICOM metadata tags may cause the sequence router to select inappropriate series.
- **Slice coverage assumptions:** The model assumes complete short-axis coverage. Studies with incomplete coverage (e.g., limited to specific levels) may produce overconfident negative statements.

---

## Bias and Ethics Considerations

### Potential Biases

- **Demographic bias:** Training population reflects Stanford Medical Center demographics. Performance on populations with different baseline cardiovascular disease prevalence is not fully validated.
- **Scanner and protocol bias:** Models trained on 1.5T and 3T studies from specific vendors may underperform on studies from other vendors or field strengths.
- **Interpretive style:** Output reflects the interpretive conventions of Stanford cardiac radiologists, including how findings are prioritized and worded.

### Clinical Safety

- MARCUS-CMR is **not an FDA-cleared medical device**.
- CMR interpretation requires clinical context (symptoms, history, indication) not available to the model. All outputs must be reviewed by a qualified physician before clinical use.
- LGE pattern interpretation (distinguishing ischemic from non-ischemic scar) has significant therapeutic implications and must not be acted upon without expert physician review.

### Privacy

- No patient data is stored or transmitted by the model weights or inference stack.
- DICOM metadata (patient demographics, study identifiers) is read only for sequence routing and is not used as model input or stored.

---

## How to Use

### CLI

```bash
# Start the CMR expert API + web UI
video-chat-cmr
# UI available at http://localhost:8765
```

### Python API

```python
import httpx
from video_chat_ui.preprocessing.cmr import process_cmr_study

# Preprocess DICOM study
process_cmr_study("cmr_study.tgz", "cmr_grid.mp4", workdir="/tmp")

# Upload and query
client = httpx.Client(base_url="http://localhost:8000")

with open("cmr_grid.mp4", "rb") as f:
    resp = client.post("/upload", files={"file": f})
media_id = resp.json()["id"]

answer = client.post("/chat", json={
    "message": "Describe any areas of late gadolinium enhancement and their distribution pattern.",
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
