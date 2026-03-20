"""Pre-loaded example data and template questions for the MARCUS demo."""

from pathlib import Path

EXAMPLES: dict[str, list[dict]] = {
    "ecg": [
        {
            "name": "ECG Sample 1",
            "path": "/home/masadi/temp_ecg.png",
            "description": "12-lead ECG — standard clinical format",
            "default_question": "What is the cardiac rhythm and rate? Are there any conduction abnormalities?",
        },
        {
            "name": "ECG Sample 2",
            "path": "/home/masadi/temp_ecg_2.png",
            "description": "12-lead ECG — second patient",
            "default_question": "Are there ST-segment or T-wave changes suggestive of ischaemia?",
        },
        {
            "name": "ECG Sample 3",
            "path": "/home/masadi/ecg/ecg_imgs/0000abaf4c815639bcf6.png",
            "description": "12-lead ECG — dataset sample",
            "default_question": "Provide a full interpretation of this ECG including rate, rhythm, axis, and intervals.",
        },
    ],
    "echo": [
        {
            "name": "Echo Sample 1",
            "path": "/home/masadi/temp_echo.mp4",
            "description": "Multi-view echocardiogram grid",
            "default_question": "How would you grade the left ventricular systolic function? Are there wall motion abnormalities?",
        },
        {
            "name": "Echo Sample 2",
            "path": "/home/masadi/demo_data/grid_videos_small/EA64637a7-EA8409d8f_grid_small.mp4",
            "description": "Echocardiogram grid — patient A",
            "default_question": "Is there any significant valvular pathology visible?",
        },
        {
            "name": "Echo Sample 3",
            "path": "/home/masadi/demo_data/grid_videos_small/EA64637a7-EA840cd84_grid_small.mp4",
            "description": "Echocardiogram grid — patient B",
            "default_question": "Describe the overall cardiac structure and function from this echocardiogram.",
        },
    ],
    "cmr": [
        {
            "name": "CMR Sample 1",
            "path": "/home/masadi/temp_cmr.mp4",
            "description": "Multi-sequence cardiac MRI grid",
            "default_question": "Describe the myocardial morphology and any late gadolinium enhancement.",
        },
        {
            "name": "CMR Sample 2",
            "path": "/home/masadi/cmr_grid.mp4",
            "description": "CMR cine grid",
            "default_question": "What is the estimated left ventricular ejection fraction and is there evidence of cardiomyopathy?",
        },
        {
            "name": "CMR Sample 3",
            "path": "/home/masadi/videos_cmr_50/EA64637b7-EA6463b9b_grid.mp4",
            "description": "CMR grid — dataset sample",
            "default_question": "Is there evidence of myocardial fibrosis or scar tissue?",
        },
    ],
}

MULTIMODAL_EXAMPLE = {
    "ecg": "/home/masadi/temp_ecg.png",
    "echo": "/home/masadi/temp_echo.mp4",
    "cmr": "/home/masadi/temp_cmr.mp4",
}

TEMPLATE_QUESTIONS: dict[str, list[str]] = {
    "ecg": [
        "What is the cardiac rhythm and rate?",
        "Are there ST-segment or T-wave changes?",
        "Is there evidence of left ventricular hypertrophy?",
        "Are there any conduction abnormalities?",
        "Provide a full interpretation of this ECG.",
    ],
    "echo": [
        "How would you grade the left ventricular systolic function?",
        "Are there any regional wall motion abnormalities?",
        "Is there any significant valvular pathology?",
        "Is there a pericardial effusion?",
        "Describe the overall findings from this echocardiogram.",
    ],
    "cmr": [
        "What is the left ventricular ejection fraction?",
        "Is there any late gadolinium enhancement?",
        "Describe the myocardial morphology and tissue characterisation.",
        "Is there evidence of cardiomyopathy?",
        "Summarise all findings from this CMR study.",
    ],
    "multimodal": [
        "Summarise this patient's cardiac findings from all available modalities.",
        "Is there evidence of ischaemic heart disease across modalities?",
        "What is the most likely unifying diagnosis integrating all data?",
        "Are there any discrepancies between modalities that warrant further investigation?",
    ],
}


def get_example_choices(modality: str) -> list[str]:
    """Return display names for the examples dropdown."""
    return [ex["name"] for ex in EXAMPLES.get(modality, [])]


def get_example(modality: str, name: str) -> dict | None:
    """Look up an example by modality and display name."""
    for ex in EXAMPLES.get(modality, []):
        if ex["name"] == name:
            return ex
    return None


def example_exists(modality: str, name: str) -> bool:
    """Check whether the example file actually exists on disk."""
    ex = get_example(modality, name)
    return ex is not None and Path(ex["path"]).is_file()
