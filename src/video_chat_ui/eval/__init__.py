"""Batch VQA/MCQ judging via OpenAI (gpt-4o-mini)."""

from video_chat_ui.eval.judge import score_mcq, score_vqa

__all__ = ["score_vqa", "score_mcq"]
