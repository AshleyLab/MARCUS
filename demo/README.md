# MARCUS demo — end-to-end inference on bundled public sample data

This folder contains a self-contained demonstration of the MARCUS pipeline on a
small set of public cardiac imaging studies. It exists so that reviewers and
new users can verify the model runs end-to-end without needing to access the
HIPAA-protected Stanford or UCSF cohorts described in the manuscript.

All bundled imaging files are pre-rendered in **MARCUS input format** —
i.e. exactly what the model ingests at inference time. No preprocessing
step is required at demo time.

## What's included

Demo data are bundled directly in this folder (≈ 1.2 MB total):

| Modality   | Source                              | n | Format (MARCUS input) |
|------------|-------------------------------------|---|-----------------------|
| 12-lead ECG | PTB-XL\* — open-access PhysioNet records | 3 | 4×3 PNG grid |
| Echocardiogram | EchoNet-Dynamic\*\* — Stanford open-access cines | 3 | MP4 single-clip cine |
| Cardiac MRI | ACDC\*\*\* — Automated Cardiac Diagnosis Challenge | 3 | MP4 multi-slice cine grid |
| Multimodal | hand-crafted pairings of the above | 3 | (combinations) |

\* Wagner P, et al. *Sci Data* 7, 154 (2020). ODC-By 1.0.
\*\* Ouyang D, et al. *Nature* 580, 252–256 (2020). Stanford Academic Software License.
\*\*\* Bernard O, et al. *IEEE TMI* 37(11), 2514–2525 (2018). CC-BY 4.0.

Full per-dataset license text is in `LICENSES/`.

## Quick start (peer reviewers)

The MARCUS model weights are hosted in a private Hugging Face repository
during peer review. To run the demo end-to-end:

```bash
# 1. Install Python dependencies
pip install torch torchvision transformers accelerate \
            qwen-vl-utils huggingface_hub safetensors decord pillow

# 2. Authenticate with the reviewer access token
#    (token provided to the editorial office; see your reviewer instructions)
export HF_TOKEN=hf_<reviewer_token_here>

# 3. Run the demo
python scripts/run_demo.py
```

Expected runtime: ≈ 3–5 minutes on a single consumer GPU. Weights
(~21 GB across three per-modality experts) download automatically on
first run and are cached afterwards.

The script prints each model output and compares it against
`expected_outputs.json`. A pass/fail summary is printed at the end.

## Folder layout

```
demo/
├── README.md
├── LICENSES/                       ← license text for each public dataset
├── data/
│   ├── ecg/{sample_001.png, sample_002.png, sample_003.png}
│   ├── echo/{sample_001.mp4, sample_002.mp4, sample_003.mp4}
│   └── cmr/{sample_001.mp4, sample_002.mp4, sample_003.mp4}
├── questions.json                  ← 15 demo Q–A pairs (5 per modality + 3 multimodal)
├── expected_outputs.json           ← reference MARCUS answers + provenance notes
└── scripts/
    └── run_demo.py                 ← end-to-end MARCUS inference + auto-verify
```

## Model weights

The trained MARCUS checkpoints (three per-modality experts: CMR, Echo, ECG)
are hosted privately at `jackosullivan/MARCUS-3B-private` on Hugging Face
during peer review. Reviewer access is granted via a read-only token shared
with the editorial office. Upon publication, the repository will be made
publicly accessible at the same URL.

## Source-data provenance

The 9 bundled imaging files are derivative preprocessed renderings of the
following public-dataset records:

| File | Original source record |
|---|---|
| `ecg/sample_001.png` | PTB-XL record 00952 |
| `ecg/sample_002.png` | PTB-XL record 03665 |
| `ecg/sample_003.png` | PTB-XL record 10950 |
| `echo/sample_001.mp4` | EchoNet-Dynamic 0X100CF05D141FF143 |
| `echo/sample_002.mp4` | EchoNet-Dynamic 0X1012703CDC1436FE |
| `echo/sample_003.mp4` | EchoNet-Dynamic 0X102CFB07F752AAE6 |
| `cmr/sample_001.mp4` | ACDC training patient014 |
| `cmr/sample_002.mp4` | ACDC training patient107 |
| `cmr/sample_003.mp4` | ACDC training patient135 |

## What this demo DOES NOT do

- Reproduce the manuscript's accuracy / Likert numbers (those require the full
  Stanford, UCSF, and public test sets — see the `evaluation/` folder in the
  main repository).
- Train any models (training scripts are in `training/`).

It only verifies that the inference pipeline runs and produces sensible
outputs on the bundled samples.

## License

Each public dataset retains its original license; see `LICENSES/` for the full
text per dataset. The wrapper code (scripts/, questions.json,
expected_outputs.json) is released under the same license as the rest of the
MARCUS repository.
