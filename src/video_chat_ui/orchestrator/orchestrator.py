"""
MARCUS Agentic Orchestrator.

Implements the multimodal orchestration layer described in the MARCUS paper
(O'Sullivan et al., 2026). The orchestrator:

1. **Decomposes** a clinical query into modality-specific sub-queries.
2. **Routes** each sub-query to the appropriate domain-expert model (ECG,
   Echo, CMR) via their OpenAI-compatible REST APIs.
3. **Probes** each expert for mirage reasoning using counterfactual queries
   (no image/video attached) and flags potentially hallucinated responses.
4. **Aggregates** expert outputs, weighting each modality's contribution by
   its per-response confidence score.
5. **Synthesises** a coherent, patient-specific diagnostic response.

Architecture
------------
The orchestrator communicates with three running expert servers:
    - ECG expert   → port 8020 (model API) / 8775 (web UI)
    - Echo expert  → port 8010 (model API) / 8770 (web UI)
    - CMR expert   → port 8000 (model API) / 8765 (web UI)

Each expert exposes an OpenAI-compatible ``/v1/chat/completions`` endpoint.

References
----------
O'Sullivan JW et al., "MARCUS: An agentic, multimodal vision-language model
for cardiac diagnosis and management." (2026).
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

import httpx

from video_chat_ui.orchestrator.mirage import MirageProbe, ProbeResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Expert endpoint registry
# ---------------------------------------------------------------------------

#: Default mapping of modality labels to API and UI base URLs.
#: API ports match the LLaMA-Factory servers started by ``cli.py``.
EXPERT_ENDPOINTS: dict[str, dict[str, str]] = {
    "ecg": {
        "api_url": "http://127.0.0.1:8020",
        "ui_url": "http://127.0.0.1:8775",
        "media_kind": "image",
    },
    "echo": {
        "api_url": "http://127.0.0.1:8010",
        "ui_url": "http://127.0.0.1:8770",
        "media_kind": "video",
    },
    "cmr": {
        "api_url": "http://127.0.0.1:8000",
        "ui_url": "http://127.0.0.1:8765",
        "media_kind": "video",
    },
}

# ---------------------------------------------------------------------------
# Media reference resolution
# ---------------------------------------------------------------------------


def _resolve_media_ref(media_id: str, media_kind: str, ui_url: str) -> str:
    """Return the appropriate media reference for the LLaMA-Factory API.

    Videos must be local file paths (the av decoder needs seekable files).
    Images can be served via HTTP URL.
    """
    if media_kind == "video":
        from video_chat_ui import config
        from pathlib import Path

        local = Path(config.UPLOAD_DIR) / media_id
        if local.is_file():
            return str(local.resolve())
    return f"{ui_url}/media/{media_id}"


# ---------------------------------------------------------------------------
# Keyword heuristics for modality selection
# ---------------------------------------------------------------------------

_MODALITY_KEYWORDS: dict[str, list[str]] = {
    "ecg": [
        "ecg", "electrocardiogram", "ekg", "rhythm", "arrhythmia",
        "qrs", "p-wave", "t-wave", "st segment", "st elevation",
        "st depression", "pr interval", "qt interval", "atrial fibrillation",
        "afib", "ventricular tachycardia", "vt", "bradycardia", "tachycardia",
        "heart block", "bundle branch", "lbbb", "rbbb", "voltage",
        "hypertrophy", "ischemia", "infarction", "repolarisation",
        "repolarization", "j wave", "delta wave", "wpw",
    ],
    "echo": [
        "echo", "echocardiogram", "echocardiography", "ultrasound",
        "wall motion", "ejection fraction", "ef", "mitral", "aortic",
        "tricuspid", "pulmonary valve", "diastol", "systol",
        "left ventricle", "lv", "right ventricle", "rv", "atrium",
        "septal", "doppler", "e/a ratio", "tissue doppler",
        "valvular", "regurgitation", "stenosis", "pericardial effusion",
        "tamponade", "cardiomyopathy", "hypertrophic", "dilated",
        "apical", "parasternal",
    ],
    "cmr": [
        "cmr", "cardiac mri", "cardiac magnetic resonance", "mri",
        "late gadolinium", "lge", "fibrosis", "scar", "myocarditis",
        "t2", "t1 mapping", "cine", "short axis", "long axis",
        "pericardial", "pericarditis", "constrictive",
        "arrhythmogenic", "arvc", "amyloid", "iron overload",
        "edema", "oedema", "infarct", "wall thickness",
    ],
}


def _select_relevant_modalities(
    question: str, available: list[str]
) -> list[str]:
    """
    Return which of *available* modalities are needed to answer *question*.

    Uses keyword matching; returns all available modalities when none match
    (conservative fallback for general or ambiguous queries).
    """
    q_lower = question.lower()
    selected = [
        mod
        for mod in available
        if any(kw in q_lower for kw in _MODALITY_KEYWORDS.get(mod, []))
    ]
    # If no keywords matched, assume all modalities are relevant
    return selected if selected else list(available)


# ---------------------------------------------------------------------------
# Sub-query decomposition
# ---------------------------------------------------------------------------

_DECOMPOSE_SYSTEM_PROMPT = (
    "You are a clinical cardiology AI assistant helping decompose complex "
    "multimodal diagnostic queries into modality-specific sub-questions. "
    "Given a question and a list of available imaging modalities, output "
    "one focused sub-question per modality that together address the original "
    "question. Keep each sub-question concise (one sentence)."
)


async def _decompose_query_with_llm(
    question: str,
    modalities: list[str],
    api_url: str,
    model: str,
    timeout: float,
) -> dict[str, str]:
    """
    Use the CMR/first available expert LLM to decompose the query.

    Falls back to a simple heuristic if the LLM call fails.
    """
    mod_list = ", ".join(modalities)
    user_msg = (
        f"Available modalities: {mod_list}\n\n"
        f"Clinical question: {question}\n\n"
        f"Generate one focused sub-question for each modality. "
        f"Format your response as:\n"
        + "\n".join(f"{m.upper()}: <sub-question>" for m in modalities)
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _DECOMPOSE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.1,
        "max_tokens": 300,
        "stream": False,
    }
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{api_url}/v1/chat/completions",
                json=payload,
                timeout=timeout,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        logger.warning("LLM decomposition failed, using heuristic: %s", exc)
        return {m: question for m in modalities}

    # Parse "ECG: <sub-question>" lines
    result: dict[str, str] = {}
    for line in text.splitlines():
        for mod in modalities:
            prefix = f"{mod.upper()}:"
            if line.strip().upper().startswith(prefix):
                result[mod] = line.strip()[len(prefix) :].strip()
                break
    # Fill any missing modalities with the original question
    for mod in modalities:
        if mod not in result:
            result[mod] = question
    return result


# ---------------------------------------------------------------------------
# Synthesis prompt
# ---------------------------------------------------------------------------

_SYNTHESIS_SYSTEM_PROMPT = (
    "You are MARCUS, a multimodal cardiovascular AI. You have received "
    "independent assessments from specialist ECG, echocardiography, and "
    "cardiac MRI expert models. Synthesise their findings into a single, "
    "coherent clinical summary that directly answers the clinician's question. "
    "Integrate complementary information across modalities. Flag any conflicts. "
    "Be concise and clinically precise."
)


# ---------------------------------------------------------------------------
# Main result dataclass
# ---------------------------------------------------------------------------

@dataclass
class OrchestratorResult:
    """Full result returned by :meth:`MARCUSOrchestrator.synthesize`."""

    question: str
    answer: str
    modality_responses: dict[str, str] = field(default_factory=dict)
    sub_questions: dict[str, str] = field(default_factory=dict)
    probe_results: dict[str, ProbeResult] = field(default_factory=dict)
    confidence_scores: dict[str, float] = field(default_factory=dict)
    mirage_flags: dict[str, bool] = field(default_factory=dict)
    flagged_modalities: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# MARCUSOrchestrator
# ---------------------------------------------------------------------------

class MARCUSOrchestrator:
    """
    Agentic orchestrator for multimodal cardiac diagnosis with MARCUS.

    Parameters
    ----------
    expert_endpoints : dict
        Mapping of ``{modality: {"api_url": ..., "ui_url": ..., "media_kind": ...}}``.
        Defaults to :data:`EXPERT_ENDPOINTS`.
    enable_mirage_probing : bool
        Whether to run counterfactual mirage probing for each expert response.
        Disable for faster inference when mirage detection is not required.
    mirage_threshold : float
        Similarity threshold for mirage flagging (passed to :class:`MirageProbe`).
    model : str
        Model name string forwarded to the OpenAI-compatible ``/v1/chat/completions``
        endpoint. Typically ``"default"`` for LLaMA-Factory.
    temperature : float
        Sampling temperature for all expert and synthesis calls.
    max_tokens : int
        Maximum tokens for expert sub-query responses.
    synthesis_max_tokens : int
        Maximum tokens for the final synthesis response.
    timeout : float
        Per-request HTTP timeout in seconds.
    use_llm_decomposition : bool
        If True, use an LLM to decompose queries; otherwise use keyword heuristics.
    """

    def __init__(
        self,
        expert_endpoints: Optional[dict[str, dict[str, str]]] = None,
        enable_mirage_probing: bool = True,
        mirage_threshold: float = 0.85,
        model: str = "default",
        temperature: float = 0.1,
        max_tokens: int = 512,
        synthesis_max_tokens: int = 1024,
        timeout: float = 120.0,
        use_llm_decomposition: bool = False,
    ) -> None:
        self.expert_endpoints = expert_endpoints or EXPERT_ENDPOINTS
        self.enable_mirage_probing = enable_mirage_probing
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.synthesis_max_tokens = synthesis_max_tokens
        self.timeout = timeout
        self.use_llm_decomposition = use_llm_decomposition

        self._mirage_probe = MirageProbe(
            similarity_threshold=mirage_threshold,
            rephrase_count=3,
            timeout=timeout,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _available_experts(self, media_ids: dict[str, str]) -> list[str]:
        """Return expert names for which media has been supplied."""
        return [
            mod
            for mod in self.expert_endpoints
            if mod in media_ids
        ]

    async def _decompose_query(
        self,
        question: str,
        relevant_modalities: list[str],
    ) -> dict[str, str]:
        """
        Decompose *question* into one sub-query per relevant modality.

        Uses either the LLM-based decomposer or simple keyword heuristics.
        """
        if not self.use_llm_decomposition or not relevant_modalities:
            return {mod: question for mod in relevant_modalities}

        # Use the first available expert as the decomposition LLM
        first_mod = relevant_modalities[0]
        api_url = self.expert_endpoints[first_mod]["api_url"]
        return await _decompose_query_with_llm(
            question, relevant_modalities, api_url, self.model, self.timeout
        )

    async def _query_expert_with_probe(
        self,
        expert: str,
        sub_question: str,
        media_id: str,
    ) -> tuple[str, ProbeResult]:
        """
        Query a single expert and optionally run mirage probing.

        Returns (response_text, probe_result).
        """
        cfg = self.expert_endpoints[expert]
        api_url = cfg["api_url"]
        ui_url = cfg["ui_url"]
        media_kind = cfg["media_kind"]

        if self.enable_mirage_probing:
            probe = await self._mirage_probe.probe_expert(
                question=sub_question,
                media_id=media_id,
                expert_api_url=api_url,
                expert=expert,
                media_kind=media_kind,
                media_base_url=ui_url,
            )
            # Use the first grounded response as the canonical answer
            response = (
                probe.image_present_responses[0]
                if probe.image_present_responses
                else ""
            )
            return response, probe
        else:
            # Lightweight path: single API call, no probing
            content: list[dict] = [{"type": "text", "text": sub_question}]
            media_ref = _resolve_media_ref(media_id, media_kind, ui_url)
            if media_kind == "video":
                content.append({"type": "video_url", "video_url": {"url": media_ref}})
            else:
                content.append({"type": "image_url", "image_url": {"url": media_ref}})

            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": content}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": False,
            }
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        f"{api_url}/v1/chat/completions",
                        json=payload,
                        timeout=self.timeout,
                    )
                    resp.raise_for_status()
                    response = resp.json()["choices"][0]["message"]["content"].strip()
            except Exception as exc:
                logger.error("Expert %s call failed: %s", expert, exc)
                response = ""

            # Create a minimal probe result (no probing performed)
            probe = ProbeResult(
                expert=expert,
                question=sub_question,
                media_id=media_id,
                media_kind=media_kind,
                image_present_responses=[response],
                image_absent_response="",
                consistency_score=1.0,
                divergence_score=1.0,
                confidence_score=1.0,
                mirage_flag=False,
            )
            return response, probe

    async def _synthesise_responses(
        self,
        question: str,
        modality_responses: dict[str, str],
        confidence_scores: dict[str, float],
        flagged_modalities: list[str],
    ) -> str:
        """
        Combine expert outputs into a single coherent answer using an LLM.

        Weights each modality response by its confidence score and notes
        any flagged (potentially mirage-affected) modalities.
        """
        if not modality_responses:
            return "No expert responses were available for synthesis."

        # Build a structured context string
        context_lines = []
        for mod, resp in modality_responses.items():
            conf = confidence_scores.get(mod, 1.0)
            flag = " [⚠ low confidence — possible mirage]" if mod in flagged_modalities else ""
            context_lines.append(
                f"[{mod.upper()} Expert (confidence {conf:.2f}){flag}]\n{resp}"
            )
        context = "\n\n".join(context_lines)

        user_msg = (
            f"Clinical question: {question}\n\n"
            f"Expert model findings:\n{context}\n\n"
            "Please synthesise these findings into a comprehensive clinical answer."
        )

        # Use the first available expert API for synthesis
        first_mod = next(iter(modality_responses))
        api_url = self.expert_endpoints[first_mod]["api_url"]

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYNTHESIS_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "temperature": self.temperature,
            "max_tokens": self.synthesis_max_tokens,
            "stream": False,
        }
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{api_url}/v1/chat/completions",
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            logger.error("Synthesis LLM call failed: %s", exc)
            # Graceful degradation: concatenate expert responses directly
            return "\n\n".join(
                f"**{mod.upper()}**: {resp}"
                for mod, resp in modality_responses.items()
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def synthesize(
        self,
        question: str,
        media_ids: dict[str, str],
        messages_history: Optional[list[dict]] = None,
        route_all: bool = False,
    ) -> OrchestratorResult:
        """
        Full multimodal inference pipeline.

        Parameters
        ----------
        question : str
            Free-text clinical question (may reference multiple modalities).
        media_ids : dict[str, str]
            Mapping of ``{modality: media_file_id}`` for each available
            modality. Media file IDs are returned by the preprocessing
            endpoints (``/upload`` or ``/preprocess``). Example::

                {"ecg": "abc123", "echo": "def456", "cmr": "ghi789"}

        messages_history : list[dict] | None
            Optional prior conversation turns (OpenAI format). Currently
            passed through but not used in sub-query routing.

        Returns
        -------
        OrchestratorResult
            Structured result containing the synthesised answer, per-modality
            responses, confidence scores, probe results, and mirage flags.
        """
        available = self._available_experts(media_ids)
        if not available:
            return OrchestratorResult(
                question=question,
                answer="No media was provided. Please supply at least one cardiac study.",
            )

        # 1. Select relevant modalities
        # When route_all is True (multimodal tab), query every provided expert
        relevant = available if route_all else _select_relevant_modalities(question, available)
        logger.info(
            "Query: %r | Available: %s | Relevant: %s",
            question[:80],
            available,
            relevant,
        )

        # 2. Decompose query
        sub_questions = await self._decompose_query(question, relevant)

        # 3. Query all relevant experts concurrently
        tasks = {
            mod: self._query_expert_with_probe(
                mod, sub_questions[mod], media_ids[mod]
            )
            for mod in relevant
        }
        expert_results = dict(
            zip(tasks.keys(), await asyncio.gather(*tasks.values()))
        )

        # 4. Collect results
        modality_responses: dict[str, str] = {}
        probe_results: dict[str, ProbeResult] = {}
        confidence_scores: dict[str, float] = {}
        mirage_flags: dict[str, bool] = {}
        flagged_modalities: list[str] = []

        for mod, (response, probe) in expert_results.items():
            modality_responses[mod] = response
            probe_results[mod] = probe
            confidence_scores[mod] = probe.confidence_score
            mirage_flags[mod] = probe.mirage_flag
            if probe.mirage_flag:
                flagged_modalities.append(mod)
                logger.warning(
                    "Modality %s flagged as potential mirage (confidence=%.3f)",
                    mod,
                    probe.confidence_score,
                )

        # 5. Synthesise final answer
        answer = await self._synthesise_responses(
            question, modality_responses, confidence_scores, flagged_modalities
        )

        return OrchestratorResult(
            question=question,
            answer=answer,
            modality_responses=modality_responses,
            sub_questions=sub_questions,
            probe_results=probe_results,
            confidence_scores=confidence_scores,
            mirage_flags=mirage_flags,
            flagged_modalities=flagged_modalities,
        )
