"""
MARCUS GRPO Reward Function (verl-compatible).

Implements rule-based reward for MCQ fine-tuning used in the Stage 3 GRPO
training via the `verl` framework. The reward is computed by the
``compute_score`` function, which verl calls per generated response.

Reward signal:
  - 1.0  : model output matches ground truth MCQ answer
  - 0.0  : incorrect or unparseable response

The ground truth is a full answer string such as ``"B. Mild LV dilation"``
and the model is expected to reproduce or contain the correct answer letter
and/or text.

Answer matching uses ``mathruler.grader.grade_answer`` which handles
partial string matching, letter extraction, and normalisation.

Integration with verl
---------------------
This module is registered in ``verl/verl/utils/reward_score/__init__.py``
via the ``data_source`` field in the training parquet files:

.. code-block:: python

    # In verl/verl/utils/reward_score/__init__.py
    elif 'echo' in data_source:
        from . import echo_mcq
        res = echo_mcq.compute_score(solution_str, ground_truth)
    elif 'ecg' in data_source:
        from . import ecg_simple
        res = ecg_simple.compute_score(solution_str, ground_truth)
    # CMR routes similarly via data_source path matching

Training details (from MARCUS paper, O'Sullivan et al. 2026):
  - GRPO applied after SFT on MCQ datasets:
      Echo: cardiology_mcq_echo_train.json (~9,000 sampled examples)
      CMR:  cardiology_mcq_cmr_train.json  (~8,192 sampled examples)
      ECG:  ecg_diag_train.json            (~4,000 sampled examples)
  - Group size (rollout n): 4 responses per prompt
  - KL loss coefficient: 0.01 (low_var_kl type)
  - Learning rate: 1e-6

References
----------
Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in
Open Language Models." arXiv:2402.03300 (2024). [GRPO algorithm]

O'Sullivan JW et al., "MARCUS: An agentic, multimodal vision-language model
for cardiac diagnosis and management." (2026).
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Answer extraction (fallback when mathruler is unavailable)
# ---------------------------------------------------------------------------

_PATTERNS: list[re.Pattern] = [
    # Explicit label: "The answer is A" / "Answer: B"
    re.compile(r"(?:the\s+)?answer\s+is\s*:?\s*([A-E])\b", re.IGNORECASE),
    # "A is correct" / "option A" etc.
    re.compile(r"\b(?:option\s+)?([A-E])\s+is\s+(?:correct|the\s+answer)\b", re.IGNORECASE),
    # Standalone letter at start of response: "A." / "B)" / "C "
    re.compile(r"^\s*([A-E])[.):\s]", re.IGNORECASE | re.MULTILINE),
    # Final letter in a reasoning chain
    re.compile(r"therefore[,\s]+(?:the\s+)?(?:answer|choice|option)\s+(?:is\s+)?([A-E])\b", re.IGNORECASE),
    # Bare letter as the entire response
    re.compile(r"^\s*([A-E])\s*$", re.IGNORECASE),
    # Any isolated letter A-E (last resort)
    re.compile(r"\b([A-E])\b", re.IGNORECASE),
]


def _extract_letter(text: str) -> Optional[str]:
    """Extract an MCQ answer letter (A–E) from model output text."""
    if not text or not text.strip():
        return None
    for pat in _PATTERNS:
        m = pat.search(text)
        if m:
            letter = m.group(1).upper()
            if letter in "ABCDE":
                return letter
    return None


def _letter_matches(predict_str: str, ground_truth: str) -> bool:
    """Check whether the predicted response matches the ground truth answer.

    Ground truth is typically the full answer text, e.g. "B. Mild LV dilation".
    Matching checks:
      1. Whether the ground truth string appears in the prediction (case-insensitive)
      2. Whether extracted letters match
    """
    gt = ground_truth.strip()
    pred = predict_str.strip()

    # Direct containment (case-insensitive)
    if gt.lower() in pred.lower():
        return True

    # Letter-based matching: extract letter from GT and prediction
    gt_letter: Optional[str] = None
    if gt and gt[0].upper() in "ABCDE" and (len(gt) == 1 or gt[1] in ". )"):
        gt_letter = gt[0].upper()

    if gt_letter is None:
        gt_letter = _extract_letter(gt)

    pred_letter = _extract_letter(pred)
    # Also check the last line (verl echo_mcq uses last line as answer)
    last_line = pred.split("\n")[-1].strip()
    if pred_letter is None:
        pred_letter = _extract_letter(last_line)

    if gt_letter and pred_letter:
        return gt_letter == pred_letter

    return False


# ---------------------------------------------------------------------------
# verl-compatible compute_score entry point
# ---------------------------------------------------------------------------

def compute_score(predict_str: str, ground_truth: str, **kwargs) -> float:
    """Compute reward for a single GRPO training example.

    This is the function registered in verl's reward_score module and called
    by the verl GRPO trainer during rollout scoring.

    Parameters
    ----------
    predict_str : str
        Model's generated response text.
    ground_truth : str
        Correct answer — typically the full answer text starting with a letter,
        e.g. ``"B. Mild left ventricular dilation"``.

    Returns
    -------
    float
        1.0 for correct, 0.0 for incorrect or unparseable.
    """
    try:
        # Prefer mathruler grader if available (used by verl in production)
        from mathruler.grader import grade_answer
        # verl echo_mcq.py uses the last \n-separated line as the answer
        answer = predict_str.split("\n")[-1]
        return 1.0 if grade_answer(answer, ground_truth) else 0.0
    except ImportError:
        pass

    # Fallback: regex-based letter matching
    correct = _letter_matches(predict_str, ground_truth)
    reward = 1.0 if correct else 0.0
    logger.debug("Reward %.1f | gt=%r | response=%r", reward, ground_truth[:60], predict_str[:60])
    return reward


# ---------------------------------------------------------------------------
# CLI / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    test_cases = [
        # (response, ground_truth, expected_reward)
        ("B. Mild left ventricular dilation", "B. Mild left ventricular dilation", 1.0),
        ("A. Sinus rhythm", "B. Atrial fibrillation", 0.0),
        ("The answer is C\nC. Normal EF", "C. Normal EF", 1.0),
        ("D. The ejection fraction is normal.", "D. Normal ejection fraction", 1.0),
        ("I cannot determine the answer from this image.", "A. Sinus rhythm", 0.0),
    ]

    print("MARCUS GRPO Reward — smoke test")
    print("-" * 50)
    all_pass = True
    for resp, gt, expected in test_cases:
        reward = compute_score(resp, gt)
        status = "PASS" if abs(reward - expected) < 1e-6 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"[{status}]  gt={gt!r:.40}  expected={expected}  got={reward}")

    sys.exit(0 if all_pass else 1)
