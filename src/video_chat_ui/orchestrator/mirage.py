"""
MARCUS Mirage Reasoning Detection via Counterfactual Probing.

Implements the three-step verification procedure described in the MARCUS paper
(O'Sullivan et al., 2026) to detect "mirage reasoning" — the phenomenon
whereby vision-language models generate plausible clinical descriptions without
actually referencing the provided image or video.

Detection protocol (per modality, per query):
    1. Generate three semantically equivalent but syntactically distinct
       rephrasings of the clinical sub-query.
    2. Route all three rephrasings WITH the image/video to the expert model
       and record responses.
    3. Issue an image-ABSENT version of the same query as a counterfactual
       probe to establish the model's language prior.
    4. Compute a per-modality confidence score from:
       - Consistency among the three image-present responses (high = grounded)
       - Divergence between image-present responses and image-absent baseline
         (high = grounded, low = potential mirage)
    5. Flag as mirage if divergence score falls below threshold.

References:
    - O'Sullivan et al. "MARCUS: An agentic, multimodal vision-language model
      for cardiac diagnosis and management." (2026).
    - See companion paper on mirage reasoning for full methodology.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rephrase templates — semantically equivalent question variations
# ---------------------------------------------------------------------------

_REPHRASE_TEMPLATES = [
    # Template 0: direct question form (original, unchanged)
    "{question}",
    # Template 1: "please describe" framing
    "Please describe {subject} as seen in the provided {modality} data.",
    # Template 2: "I would like to know" framing
    "I would like to know: {question} Please be specific.",
    # Template 3: third-person clinical framing
    "The clinician asks: {question}",
    # Template 4: imperative form
    "Based on the {modality} provided, answer the following: {question}",
]

_MODALITY_LABELS = {
    "ecg": "ECG",
    "echo": "echocardiogram",
    "cmr": "cardiac MRI",
}


def _simple_rephrase(question: str, modality: str, index: int) -> str:
    """Generate the i-th rephrase of a question for a given modality."""
    template = _REPHRASE_TEMPLATES[index % len(_REPHRASE_TEMPLATES)]
    mod_label = _MODALITY_LABELS.get(modality, "imaging")
    # Derive a crude subject noun phrase from the question
    subject = re.sub(r"^(what is |what are |describe |is there |are there )", "", question, flags=re.IGNORECASE).rstrip("?.")
    return template.format(question=question, subject=subject, modality=mod_label)


# ---------------------------------------------------------------------------
# Token-overlap similarity (Jaccard on word tokens)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    """Lowercase word tokenization."""
    return set(re.findall(r"\b[a-z]+\b", text.lower()))


def _jaccard(a: str, b: str) -> float:
    """Jaccard similarity between two strings, token-level."""
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta and not tb:
        return 1.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Data class for probe results
# ---------------------------------------------------------------------------

@dataclass
class ProbeResult:
    """Result of a mirage probe for a single expert model and sub-query."""

    expert: str
    question: str
    media_id: Optional[str]
    media_kind: str

    # Responses to 3 rephrased queries WITH image/video
    image_present_responses: list[str] = field(default_factory=list)

    # Response to image-ABSENT counterfactual probe
    image_absent_response: str = ""

    # Derived scores
    consistency_score: float = 0.0   # How similar are the 3 grounded answers?
    divergence_score: float = 0.0    # How different are grounded vs. non-grounded?
    confidence_score: float = 0.0    # Combined: high = grounded, low = mirage risk

    mirage_flag: bool = False        # True → likely mirage


# ---------------------------------------------------------------------------
# MirageProbe
# ---------------------------------------------------------------------------

class MirageProbe:
    """
    Counterfactual mirage-probing for MARCUS expert models.

    Sends each clinical sub-query to an expert API endpoint three times with
    different phrasings (grounded) and once without any image (counterfactual),
    then scores the resulting responses to detect mirage reasoning.

    Parameters
    ----------
    similarity_threshold : float
        Divergence score *below* which a response is flagged as mirage.
        Default 0.85 (responses must differ from image-absent baseline by at
        least 1 - 0.85 = 15% in token overlap to be considered grounded).
    rephrase_count : int
        Number of syntactically distinct rephrasings to generate (default 3).
    timeout : float
        Per-request timeout in seconds (default 60).
    model : str
        Model name to pass to the OpenAI-compatible API (default "default").
    temperature : float
        Sampling temperature for expert API calls.
    max_tokens : int
        Maximum tokens per expert response.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        rephrase_count: int = 3,
        timeout: float = 60.0,
        model: str = "default",
        temperature: float = 0.1,
        max_tokens: int = 256,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.rephrase_count = rephrase_count
        self.timeout = timeout
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        question: str,
        media_id: Optional[str],
        media_kind: str,
        base_url: str,
    ) -> list[dict]:
        """Build an OpenAI-compatible messages list, optionally with media."""
        content: list[dict] = [{"type": "text", "text": question}]
        if media_id is not None:
            media_ref = self._resolve_media_ref(media_id, media_kind, base_url)
            if media_kind == "video":
                content.append({"type": "video_url", "video_url": {"url": media_ref}})
            else:
                content.append({"type": "image_url", "image_url": {"url": media_ref}})
        return [{"role": "user", "content": content}]

    @staticmethod
    def _resolve_media_ref(media_id: str, media_kind: str, base_url: str) -> str:
        """Videos need local paths (av needs seekable files); images use HTTP."""
        if media_kind == "video":
            from video_chat_ui import config
            from pathlib import Path
            local = Path(config.UPLOAD_DIR) / media_id
            if local.is_file():
                return str(local.resolve())
        return f"{base_url}/media/{media_id}"

    async def _call_expert(
        self,
        client: httpx.AsyncClient,
        api_url: str,
        messages: list[dict],
    ) -> str:
        """Make one non-streaming chat completion call to an expert API."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        try:
            resp = await client.post(
                f"{api_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            logger.warning("Expert API call failed (%s): %s", api_url, exc)
            return ""

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _compute_consistency(self, responses: list[str]) -> float:
        """
        Mean pairwise Jaccard similarity among the rephrased-query responses.

        High consistency → the model gives coherent grounded answers.
        """
        if len(responses) < 2:
            return 1.0
        pairs = [
            _jaccard(responses[i], responses[j])
            for i in range(len(responses))
            for j in range(i + 1, len(responses))
        ]
        return sum(pairs) / len(pairs) if pairs else 1.0

    def _compute_divergence(
        self, with_image: list[str], without_image: str
    ) -> float:
        """
        1 - mean similarity between image-present responses and image-absent baseline.

        High divergence (close to 1) → responses changed substantially when the
        image was provided, indicating genuine visual grounding.
        Low divergence (close to 0) → responses are similar with and without the
        image, suggesting mirage reasoning.
        """
        if not with_image or not without_image:
            return 0.0
        sims = [_jaccard(r, without_image) for r in with_image]
        mean_sim = sum(sims) / len(sims)
        return 1.0 - mean_sim  # High divergence = not similar = grounded

    def _is_mirage(self, divergence: float) -> bool:
        """Return True when divergence is below the similarity threshold.

        Equivalently: mirage if grounded responses are *too similar* to the
        image-absent baseline.
        """
        # divergence = 1 - similarity; mirage when similarity is HIGH
        similarity = 1.0 - divergence
        return similarity > self.similarity_threshold

    def _compute_confidence(
        self, consistency: float, divergence: float
    ) -> float:
        """
        Composite confidence score (0–1).

        Combines consistency (are the three grounded answers coherent?) and
        divergence (do they differ from the image-absent baseline?).
        Equal weighting; both must be high for high confidence.
        """
        return 0.5 * consistency + 0.5 * divergence

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rephrase_question(self, question: str, modality: str = "echo") -> list[str]:
        """
        Generate ``rephrase_count`` syntactically distinct but semantically
        equivalent versions of *question*.

        Parameters
        ----------
        question : str
            Original clinical sub-query.
        modality : str
            Target modality label used to fill template placeholders.

        Returns
        -------
        list[str]
            List of rephrased questions (first element is always the original).
        """
        rephrases = [question]  # Always keep original as first
        for i in range(1, self.rephrase_count):
            rephrases.append(_simple_rephrase(question, modality, i))
        return rephrases

    async def probe_expert(
        self,
        question: str,
        media_id: Optional[str],
        expert_api_url: str,
        expert: str = "echo",
        media_kind: str = "video",
        messages_history: Optional[list[dict]] = None,
        media_base_url: Optional[str] = None,
    ) -> ProbeResult:
        """
        Run the full counterfactual probing pipeline for one expert.

        Parameters
        ----------
        question : str
            Clinical sub-query directed at this expert.
        media_id : str | None
            Media file ID served at ``{expert_api_url}/media/{media_id}``.
            Pass None to disable media attachment (pure text query).
        expert_api_url : str
            Base URL of the expert's OpenAI-compatible API
            (e.g. ``http://127.0.0.1:8000``).
        expert : str
            Expert name label ("ecg", "echo", "cmr"). Used for rephrasing.
        media_kind : str
            ``"video"`` or ``"image"``.
        messages_history : list[dict] | None
            Optional prior conversation context.

        Returns
        -------
        ProbeResult
            Full probe result including responses, scores, and mirage flag.
        """
        result = ProbeResult(
            expert=expert,
            question=question,
            media_id=media_id,
            media_kind=media_kind,
        )

        rephrases = self.rephrase_question(question, modality=expert)
        _media_url_base = media_base_url or expert_api_url

        async with httpx.AsyncClient() as client:
            # ----------------------------------------------------------
            # Step 1: Query with image present (3 rephrased versions)
            # ----------------------------------------------------------
            grounded_tasks = [
                self._call_expert(
                    client,
                    expert_api_url,
                    self._build_messages(q, media_id, media_kind, _media_url_base),
                )
                for q in rephrases
            ]
            result.image_present_responses = list(
                await asyncio.gather(*grounded_tasks)
            )

            # ----------------------------------------------------------
            # Step 2: Counterfactual — query WITHOUT image
            # ----------------------------------------------------------
            result.image_absent_response = await self._call_expert(
                client,
                expert_api_url,
                self._build_messages(question, None, media_kind, _media_url_base),
            )

        # ------------------------------------------------------------------
        # Step 3: Score
        # ------------------------------------------------------------------
        result.consistency_score = self._compute_consistency(
            result.image_present_responses
        )
        result.divergence_score = self._compute_divergence(
            result.image_present_responses, result.image_absent_response
        )
        result.confidence_score = self._compute_confidence(
            result.consistency_score, result.divergence_score
        )
        result.mirage_flag = self._is_mirage(result.divergence_score)

        if result.mirage_flag:
            logger.warning(
                "Mirage detected for expert=%s question=%r "
                "(divergence=%.3f, consistency=%.3f)",
                expert,
                question[:80],
                result.divergence_score,
                result.consistency_score,
            )

        return result
