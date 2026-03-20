# MARCUS Evaluation Framework

This document describes the evaluation methodology used in the MARCUS paper, including the benchmark structure, quality-filtering protocol, statistical testing approach, mirage evaluation, and step-by-step instructions for reproducing the paper results.

---

## Table of Contents

- [Evaluation Framework Overview](#evaluation-framework-overview)
- [Benchmark Dataset](#benchmark-dataset)
- [Question Formats](#question-formats)
- [B-Clean Protocol](#b-clean-protocol)
- [Evaluation Metrics](#evaluation-metrics)
- [Statistical Tests](#statistical-tests)
- [Mirage Evaluation Protocol](#mirage-evaluation-protocol)
- [Reproducing Paper Results](#reproducing-paper-results)
- [Evaluation Tools Reference](#evaluation-tools-reference)

---

## Evaluation Framework Overview

MARCUS is evaluated using a two-pronged framework: multiple-choice questions (MCQ) for measuring diagnostic accuracy in a controlled setting, and visual question answering (VQA) for assessing free-text response quality with a clinician-level judge. Both formats are applied to the same underlying clinical cases but probe complementary aspects of model capability.

```
                     MARCUS-Benchmark (1.6M questions)
                              │
              ┌───────────────┴───────────────┐
              │                               │
        MCQ Format                       VQA Format
   4-choice questions              Open-ended clinical questions
   Ground truth: correct option    Ground truth: physician report
              │                               │
              ▼                               ▼
    Exact-match + GPT judge           GPT-4-class judge
    (Correct / Incorrect / Excluded)  Likert score 1–5
              │                               │
              ▼                               ▼
       Accuracy (%)                    Mean Likert score
   (excluding Excluded items)
```

---

## Benchmark Dataset

The MARCUS-Benchmark is the largest publicly available cardiac VLM benchmark.

| Property              | Value                                                                                       |
|-----------------------|---------------------------------------------------------------------------------------------|
| Total questions       | 1,600,000+                                                                                  |
| Modalities            | ECG, Echocardiography, CMR                                                                  |
| Formats               | VQA (free-text) and MCQ (4-choice)                                                          |
| Evaluation sets       | Stanford (internal), UCSF (external)                                                        |
| Questions per modality | ~500K ECG, ~550K Echo, ~550K CMR                                                           |
| HuggingFace           | [stanford-cardiac-ai/MARCUS-Benchmark](https://huggingface.co/datasets/stanford-cardiac-ai/MARCUS-Benchmark) |

### Dataset Splits

| Split name                   | Modality | Format | Site     |
|------------------------------|----------|--------|----------|
| `ecg_mcq_stanford`           | ECG      | MCQ    | Stanford |
| `ecg_vqa_stanford`           | ECG      | VQA    | Stanford |
| `echo_mcq_stanford`          | Echo     | MCQ    | Stanford |
| `echo_mcq_ucsf`              | Echo     | MCQ    | UCSF     |
| `echo_vqa_stanford`          | Echo     | VQA    | Stanford |
| `cmr_mcq_stanford`           | CMR      | MCQ    | Stanford |
| `cmr_mcq_ucsf`               | CMR      | MCQ    | UCSF     |
| `cmr_vqa_stanford`           | CMR      | VQA    | Stanford |
| `multimodal_mcq_stanford`    | All      | MCQ    | Stanford |

### Downloading the Benchmark

```python
from datasets import load_dataset

# Load ECG MCQ questions from Stanford evaluation set
ecg_mcq = load_dataset("stanford-cardiac-ai/MARCUS-Benchmark",
                        split="ecg_mcq_stanford")

# Each item has:
# - "question":       clinical question text
# - "options":        dict {A: ..., B: ..., C: ..., D: ...}
# - "ground_truth":   correct option key ("A", "B", "C", or "D")
# - "image_path":     path to ECG PNG (or "video_path" for Echo/CMR)
# - "case_id":        anonymized case identifier
```

---

## Question Formats

### MCQ (Multiple-Choice Questions)

MCQ questions present a clinical question with four answer options (A–D). Questions are derived from physician reports by structured extraction followed by distractor generation using a physician-reviewed prompting pipeline.

#### Benchmark Augmentations

MCQ questions are generated from physician reports with plausible distractor options. The question counts reported in the paper reflect these original questions. Two augmentations were then added on top:

**1. "None of the other options" as a correct answer.**
For a subset of questions, the true diagnosis is absent from the four listed options, making "None of the other options" the correct answer. This applies to approximately 50% of Echo questions and 67% of CMR questions.

**2. Randomised option ordering.**
Answer option positions are randomised across all questions to prevent position bias.

**Example (standard):**

```
Question: What is the systolic function of the left ventricle?

A. Reduced left ventricular ejection fraction
B. Left ventricular aneurysm
C. Normal left ventricular systolic function
D. Severely impaired with global hypokinesis

Ground truth: C. Normal left ventricular systolic function
```

**Example ("None of the other options"):**

```
Question: What is the size status of the right atrium?

A. Severely dilated right atrium
B. Normal right atrial size
C. None of the other options
D. Moderately dilated right atrium

Ground truth: C. None of the other options
(True answer: Mildly dilated right atrium — not listed as an option)
```

MCQ accuracy is computed as:

```
Accuracy = (Correct) / (Correct + Incorrect)
```

Items classified as `Excluded` by the GPT judge (ambiguous question, ambiguous ground truth, or image quality insufficient for answering) are removed from both numerator and denominator.

### VQA (Visual Question Answering)

VQA questions are open-ended clinical questions requiring free-text responses. The ground truth is a physician-authored report excerpt. Model responses are evaluated by a GPT-4-class judge that assigns a Likert score from 1 to 5.

**Likert Scale:**

| Score | Meaning                                                                  |
|-------|--------------------------------------------------------------------------|
| 5     | Complete, accurate, appropriately nuanced; clinically actionable         |
| 4     | Mostly accurate with minor omissions or imprecision                      |
| 3     | Partially correct; key finding present but important details missing     |
| 2     | Substantially incomplete or containing one or more significant errors    |
| 1     | Incorrect, misleading, or hallucinatory                                  |

**Example:**

```
Question: Describe the pattern and distribution of late gadolinium enhancement.

Ground truth: "There is midwall LGE in the basal and mid lateral wall,
               consistent with non-ischemic cardiomyopathy (likely myocarditis
               or sarcoidosis pattern)."

Model answer:  "The LGE images demonstrate midwall enhancement in the
               lateral wall at the basal and mid-ventricular levels,
               without subendocardial involvement, most consistent with
               a non-ischemic scar pattern."

Likert score:  4  (accurate, minor omission of differential diagnosis)
```

---

## B-Clean Protocol

The B-Clean (Benchmark Cleaning) protocol removes questions from the evaluation set that are not reliably answerable from the provided image, ensuring that accuracy measurements reflect genuine visual reasoning rather than text-pattern matching or prior knowledge.

### Rationale

Some benchmark questions can be answered correctly without viewing the image — either because the answer is implied by the question phrasing, or because the image quality is insufficient to distinguish between plausible options. Including such questions inflates apparent accuracy for models that hallucinate plausible answers (mirage reasoning).

### B-Clean Procedure

1. **Initial question generation:** Questions are generated from physician reports using a structured prompt.

2. **Frontier model consensus check:** Each question is sent to three frontier models (GPT-5, Gemini 2.5 Pro, Claude 3.7 Sonnet) *without the image*. If all three models answer correctly (i.e., the correct answer is deducible from text alone), the question is flagged.

3. **Image quality filter:** A dedicated image quality classifier scores each input image. Questions associated with images below a quality threshold (e.g., severe motion artifact in Echo, overexposure in ECG) are flagged.

4. **Physician review sample:** A random 5% sample of flagged questions is reviewed by a board-certified cardiologist to calibrate the flagging thresholds.

5. **Exclusion:** Flagged questions are excluded from the final benchmark. The remaining set is the B-Clean benchmark.

### B-Clean Statistics

| Modality | Raw questions | Excluded | B-Clean questions | Exclusion rate |
|----------|---------------|----------|-------------------|----------------|
| ECG      | ~600K         | ~100K    | ~500K             | ~17%           |
| Echo     | ~650K         | ~100K    | ~550K             | ~15%           |
| CMR      | ~650K         | ~100K    | ~550K             | ~15%           |

---

## Evaluation Metrics

### MCQ Accuracy

```
MCQ Accuracy = |{Correct}| / (|{Correct}| + |{Incorrect}|)
```

Excluded items are removed from both the numerator and denominator.

### Mean Likert Score (VQA)

```
Mean Likert = (1/N) * sum(likert_i for i in 1..N)
```

### 95% Bootstrap Confidence Intervals

All accuracy and Likert score estimates are reported with 95% bootstrap confidence intervals. The bootstrap uses 10,000 resamples with replacement from the test set.

```python
import numpy as np

def bootstrap_ci(scores, n_bootstrap=10000, ci=0.95):
    stats = [np.mean(np.random.choice(scores, size=len(scores), replace=True))
             for _ in range(n_bootstrap)]
    lo = np.percentile(stats, (1 - ci) / 2 * 100)
    hi = np.percentile(stats, (1 + ci) / 2 * 100)
    return lo, hi
```

---

## Statistical Tests

All pairwise comparisons between MARCUS and comparator models use pre-registered statistical tests.

### McNemar Test (MCQ Pairwise Comparison)

Used to compare two models on the same set of MCQ questions. McNemar's test is appropriate for paired binary outcomes.

- **Null hypothesis:** The two models have the same proportion of correct answers.
- **Test statistic:** McNemar chi-squared on the 2×2 contingency table of (model A correct / incorrect) × (model B correct / incorrect).
- **Threshold:** p < 0.05 (two-tailed), with Bonferroni correction for multiple comparisons.

```python
from statsmodels.stats.contingency_tables import mcnemar

# contingency = [[both correct, A correct only],
#                [B correct only, both incorrect]]
result = mcnemar(contingency, exact=False, correction=True)
print(f"p-value: {result.pvalue:.4f}")
```

### Mann-Whitney U Test (VQA Likert Comparison)

Used to compare VQA Likert score distributions between two models. Mann-Whitney U is appropriate for ordinal data (Likert 1–5) that may not be normally distributed.

- **Null hypothesis:** The two models have the same Likert score distribution.
- **Test statistic:** Mann-Whitney U statistic.
- **Effect size:** Common language effect size (probability that a randomly selected MARCUS response scores higher than a randomly selected comparator response).

```python
from scipy.stats import mannwhitneyu

stat, p = mannwhitneyu(marcus_scores, comparator_scores, alternative='greater')
print(f"U={stat:.0f}, p={p:.4f}")
```

### Bootstrap Confidence Intervals for Differences

Differences in accuracy between models (e.g., MARCUS vs. GPT-5 Thinking) are reported with bootstrap confidence intervals:

```python
def bootstrap_diff_ci(scores_a, scores_b, n_bootstrap=10000):
    diffs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(scores_a), size=len(scores_a), replace=True)
        diffs.append(np.mean(scores_a[idx]) - np.mean(scores_b[idx]))
    return np.percentile(diffs, [2.5, 97.5])
```

---

## Mirage Evaluation Protocol

The mirage evaluation protocol independently measures each model's tendency to hallucinate clinically plausible but unsupported findings.

### Definition

A model response is classified as a **mirage** if:
1. The model provides a specific positive finding (e.g., "There is ST elevation in leads V1–V4") **and**
2. The same question, sent without the image, produces a similar or more confident response **and**
3. The ground truth indicates no such finding is present.

### Mirage Evaluation Procedure

1. **Image-present inference:** Send each test question with its associated image. Record the response.

2. **Image-absent inference:** Send the same question without the image (text-only). Record the response.

3. **Positive finding extraction:** A GPT-4-class extractor classifies each response as containing a positive finding (`positive`), a negative finding (`negative`), or being noncommittal (`uncertain`).

4. **Mirage classification:**
   ```
   mirage = (
       image_present_label == "positive" AND
       image_absent_label   == "positive" AND
       ground_truth_label   == "negative"
   )
   ```

5. **Mirage rate:** `mirage_rate = |{mirages}| / |{all questions with negative ground truth}|`

### MARCUS Mirage Rate

MARCUS achieves a **0% mirage rate** by design: the agentic orchestrator's counterfactual verification protocol flags and suppresses responses that produce similar outputs with and without the image. See [docs/architecture.md#mirage-resistance](architecture.md#mirage-resistance) for the implementation.

### Frontier Model Mirage Rates

| Model                     | Mirage Rate |
|---------------------------|-------------|
| MARCUS                    | 0%          |
| GPT-5 Thinking            | ~38%        |
| Gemini 2.5 Pro Deep Think | ~35%        |
| Claude 3.7 Sonnet         | ~30%        |

---

## Reproducing Paper Results

### Prerequisites

1. Install MARCUS with all extras:
   ```bash
   git clone https://github.com/masadi-99/MARCUS.git
   cd MARCUS
   pip install -e ".[all]"
   ```

2. Install and configure LLaMA-Factory:
   ```bash
   git clone https://github.com/hiyouga/LLaMA-Factory.git ~/LLaMA-Factory
   cd ~/LLaMA-Factory && pip install -e ".[torch,metrics]"
   ```

3. Download model checkpoints:
   ```bash
   python scripts/download_checkpoints.py
   ```

4. Download the benchmark:
   ```python
   from datasets import load_dataset
   for split in ["ecg_mcq_stanford", "echo_mcq_stanford", "echo_mcq_ucsf",
                 "cmr_mcq_stanford", "cmr_mcq_ucsf", "multimodal_mcq_stanford"]:
       load_dataset("stanford-cardiac-ai/MARCUS-Benchmark", split=split,
                    cache_dir="data/benchmark")
   ```

5. Set your OpenAI API key (for the GPT judge):
   ```bash
   export OPENAI_API_KEY=sk-...
   ```

### Step 1: Start Expert APIs

```bash
# In separate terminals (or use tmux/screen)
video-chat-ecg    # API on :8020
video-chat-echo   # API on :8010
video-chat-cmr    # API on :8000
```

Wait for each API to print "Model loaded" before proceeding.

### Step 2: Run Batch Inference

Run inference on each modality and evaluation split:

```bash
# ECG — Stanford
python scripts/run_inference_batch.py \
  --input   data/benchmark/ecg_mcq_stanford \
  --expert  ecg \
  --api-url http://localhost:8020 \
  --output  results/ecg_mcq_stanford_predictions.jsonl

# Echo — Stanford
python scripts/run_inference_batch.py \
  --input   data/benchmark/echo_mcq_stanford \
  --expert  echo \
  --api-url http://localhost:8010 \
  --output  results/echo_mcq_stanford_predictions.jsonl

# Echo — UCSF
python scripts/run_inference_batch.py \
  --input   data/benchmark/echo_mcq_ucsf \
  --expert  echo \
  --api-url http://localhost:8010 \
  --output  results/echo_mcq_ucsf_predictions.jsonl

# CMR — Stanford
python scripts/run_inference_batch.py \
  --input   data/benchmark/cmr_mcq_stanford \
  --expert  cmr \
  --api-url http://localhost:8000 \
  --output  results/cmr_mcq_stanford_predictions.jsonl

# CMR — UCSF
python scripts/run_inference_batch.py \
  --input   data/benchmark/cmr_mcq_ucsf \
  --expert  cmr \
  --api-url http://localhost:8000 \
  --output  results/cmr_mcq_ucsf_predictions.jsonl
```

Expected runtime: 8–24 hours per split depending on GPU (A100 recommended).

### Step 3: Run GPT Judge

```bash
for split in ecg_mcq_stanford echo_mcq_stanford echo_mcq_ucsf \
             cmr_mcq_stanford cmr_mcq_ucsf; do
  video-chat-eval \
    --input    results/${split}_predictions.jsonl \
    --task     mcq \
    --gt-key   ground_truth \
    --pred-key model_answer \
    --out-dir  results/${split}_eval/
done
```

### Step 4: Compute Statistics

```bash
for split in ecg_mcq_stanford echo_mcq_stanford echo_mcq_ucsf \
             cmr_mcq_stanford cmr_mcq_ucsf; do
  python scripts/compute_statistics.py \
    --predictions results/${split}_eval/predictions_judged.jsonl \
    --out-dir     results/${split}_stats/
done
```

Each `stats/` directory will contain:
- `accuracy.json` — MCQ accuracy with 95% bootstrap CI
- `mcnemar_vs_gpt5.json` — McNemar test vs. GPT-5 Thinking
- `mcnemar_vs_gemini.json` — McNemar test vs. Gemini 2.5 Pro Deep Think
- `summary.csv` — Combined summary table

### Step 5: VQA Evaluation (Optional)

```bash
# Run VQA inference
python scripts/run_inference_batch.py \
  --input   data/benchmark/ecg_vqa_stanford \
  --expert  ecg \
  --api-url http://localhost:8020 \
  --output  results/ecg_vqa_stanford_predictions.jsonl

# GPT judge (VQA)
video-chat-eval \
  --input    results/ecg_vqa_stanford_predictions.jsonl \
  --task     vqa \
  --gt-key   ground_truth \
  --pred-key model_answer \
  --out-dir  results/ecg_vqa_stanford_eval/
```

### Step 6: Multimodal Evaluation

The multimodal evaluation requires all three experts to be running simultaneously:

```bash
python scripts/run_inference_batch.py \
  --input     data/benchmark/multimodal_mcq_stanford \
  --expert    multimodal \
  --ecg-url   http://localhost:8020 \
  --echo-url  http://localhost:8010 \
  --cmr-url   http://localhost:8000 \
  --output    results/multimodal_mcq_stanford_predictions.jsonl
```

### Step 7: Mirage Evaluation

```bash
python scripts/run_mirage_eval.py \
  --input    data/benchmark/ecg_mcq_stanford \
  --expert   ecg \
  --api-url  http://localhost:8020 \
  --output   results/ecg_mirage_eval.jsonl
```

---

## Evaluation Tools Reference

### `video-chat-eval` (CLI)

GPT-4-class judge for VQA and MCQ evaluation.

```
Usage: video-chat-eval [OPTIONS]

Options:
  --input PATH          Input JSONL file with model predictions [required]
  --task TEXT           "vqa" or "mcq" [required]
  --out-dir PATH        Output directory [required]
  --question-key TEXT   Key for question field [default: "question"]
  --gt-key TEXT         Key for ground truth field [default: "ground_truth"]
  --pred-key TEXT       Key for model prediction field [default: "model_answer"]
  --model TEXT          OpenAI model to use as judge [default: "gpt-4o-mini"]
  --batch-size INT      Concurrent API requests [default: 10]
```

**VQA output fields added per item:**
- `likert_score` (int, 1–5)
- `likert_explanation` (str)

**MCQ output fields added per item:**
- `eval_label` ("Correct" | "Incorrect" | "Excluded")

**Summary output** (`summary.json`):
- VQA: `mean_likert`, `std_likert`, `n_items`
- MCQ: `accuracy`, `n_correct`, `n_incorrect`, `n_excluded`, `n_total`

### `run_inference_batch.py`

Runs model inference on a benchmark split.

```
Usage: python scripts/run_inference_batch.py [OPTIONS]

Options:
  --input PATH      Input benchmark directory or JSONL [required]
  --expert TEXT     "ecg" | "echo" | "cmr" | "multimodal" [required]
  --api-url URL     Expert API base URL [required for single-expert]
  --output PATH     Output JSONL file [required]
  --batch-size INT  Concurrent requests [default: 4]
  --max-items INT   Limit number of items (for testing)
```

### `compute_statistics.py`

Computes accuracy, CIs, and statistical tests from judged predictions.

```
Usage: python scripts/compute_statistics.py [OPTIONS]

Options:
  --predictions PATH    Judged predictions JSONL [required]
  --comparator PATH     Comparator model predictions (for McNemar) [optional]
  --out-dir PATH        Output directory [required]
  --n-bootstrap INT     Bootstrap resamples [default: 10000]
  --alpha FLOAT         Significance threshold [default: 0.05]
```
