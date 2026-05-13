# MARCUS demo — end-to-end inference on bundled public sample data

This folder contains a self-contained demonstration of the MARCUS pipeline on a
small set of public cardiac imaging studies. It exists so that reviewers and
new users can verify the model runs end-to-end **without downloading anything
extra** and without requiring access to the HIPAA-protected Stanford or UCSF
cohorts described in the manuscript.

All bundled files are pre-rendered in **MARCUS input format** — i.e. exactly
what the model ingests at inference time. No preprocessing step is required.

## What's included

Demo data are bundled directly in this folder (≈ 1.1 MB total):

| Modality   | Source                              | n | Format (MARCUS input) |
|------------|-------------------------------------|---|-----------------------|
| 12-lead ECG | PTB-XL\* — open-access PhysioNet records | 3 | 4×3 PNG grid (compressed waveform image) |
| Echocardiogram | EchoNet-Dynamic\*\* — Stanford open-access cines | 3 | MP4 single-clip cine |
| Cardiac MRI | ACDC\*\*\* — Automated Cardiac Diagnosis Challenge training subset | 3 | MP4 multi-slice cine grid |
| Multimodal | hand-crafted pairings of the above | 3 | (combinations) |

\* Wagner P, et al. *Sci Data* 7, 154 (2020). ODC-By 1.0.
\*\* Ouyang D, et al. *Nature* 580, 252–256 (2020). Stanford Academic Software License.
\*\*\* Bernard O, et al. *IEEE TMI* 37(11), 2514–2525 (2018). CC-BY 4.0.

Full per-dataset license text is in `LICENSES/`. Bundled files are derivative
preprocessed renderings of public-dataset source records, redistributed
under each dataset's original license terms with attribution.

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

## Quick start

```bash
# 1. (one time) install Python dependencies
pip install -r ../requirements.txt

# 2. run MARCUS on every demo question
#    Weights pulled from Hugging Face on first run (≈ 6 GB; cached afterwards)
python scripts/run_demo.py
```

Expected runtime: ≈ 3 minutes on a single consumer GPU.

The script prints each model output and compares it against
`expected_outputs.json`. A pass/fail summary is printed at the end.

## Model weights

The trained MARCUS checkpoints are hosted on Hugging Face at
`AshleyLab/MARCUS-3B` and are downloaded automatically by `run_demo.py` on
first run (≈ 6 GB; cached locally afterwards). No login required.

## Source-data provenance

The 9 bundled imaging files are derivative preprocessed renderings of the
following public-dataset records:

| File | Original source record |
|---|---|
| `ecg/sample_001.png` | PTB-XL record 00952 (records100) |
| `ecg/sample_002.png` | PTB-XL record 03665 |
| `ecg/sample_003.png` | PTB-XL record 10950 |
| `echo/sample_001.mp4` | EchoNet-Dynamic 0X100CF05D141FF143 |
| `echo/sample_002.mp4` | EchoNet-Dynamic 0X1012703CDC1436FE |
| `echo/sample_003.mp4` | EchoNet-Dynamic 0X102CFB07F752AAE6 |
| `cmr/sample_001.mp4` | ACDC training patient014 |
| `cmr/sample_002.mp4` | ACDC training patient107 |
| `cmr/sample_003.mp4` | ACDC training patient135 |

Source records can be retrieved from PhysioNet, the EchoNet-Dynamic project
page, and the ACDC challenge site respectively.

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
