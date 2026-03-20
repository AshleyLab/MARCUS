"""OpenAI Responses API judge (gpt-4o-mini)."""
from __future__ import annotations

import json
import os
import re
from typing import Any

from video_chat_ui.eval.prompts import EVAL_SYSTEM_PROMPT_MCQ, EVAL_SYSTEM_PROMPT_VQA


def _get_client():
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError("Install eval extra: pip install 'video-chat-ui[eval]'") from e
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=key)


def _parse_json_loose(text: str) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    return json.loads(text)


def score_vqa(
    question: str,
    gt: str,
    pred: str,
    *,
    model: str = "gpt-4o-mini",
) -> dict[str, Any]:
    """Return ``{"answer": 1..5, "explanation": str}`` (answer as int)."""
    client = _get_client()
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": EVAL_SYSTEM_PROMPT_VQA}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Question: "
                        + question
                        + "\n\nGround truth answer: "
                        + gt
                        + "\n\nModel answer: "
                        + pred,
                    }
                ],
            },
        ],
    )
    raw = response.output_text
    data = _parse_json_loose(raw)
    ans = data.get("answer")
    if isinstance(ans, str) and ans.isdigit():
        ans = int(ans)
    if ans not in (1, 2, 3, 4, 5):
        raise ValueError(f"Invalid VQA likert in response: {data!r}")
    return {
        "answer": ans,
        "explanation": str(data.get("explanation", "")),
    }


def score_mcq(
    question: str,
    gt: str,
    pred: str,
    *,
    model: str = "gpt-4o-mini",
) -> dict[str, Any]:
    """Return ``{"answer": "Correct"|"Incorrect"|"Excluded"}``."""
    client = _get_client()
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": EVAL_SYSTEM_PROMPT_MCQ}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Question: "
                        + question
                        + "\n\nGround truth answer: "
                        + gt
                        + "\n\nModel answer: "
                        + pred,
                    }
                ],
            },
        ],
    )
    raw = response.output_text
    data = _parse_json_loose(raw)
    ans = data.get("answer")
    if ans not in ("Correct", "Incorrect", "Excluded"):
        raise ValueError(f"Invalid MCQ label in response: {data!r}")
    return {"answer": ans}
