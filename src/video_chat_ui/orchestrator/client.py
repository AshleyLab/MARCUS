"""
MARCUS Python Client.

High-level Python API for interacting with MARCUS expert models and the
multimodal orchestrator. Wraps the preprocessing pipeline, media upload, and
chat API into a clean interface suitable for programmatic use.

Example — single expert::

    import asyncio
    from video_chat_ui.orchestrator.client import MARCUSClient

    client = MARCUSClient(expert="ecg", ui_base_url="http://localhost:8775")
    answer = asyncio.run(client.query(
        "Does this ECG show ST-segment elevation?",
        media_path="patient_ecg.npy",
    ))
    print(answer)

Example — multimodal orchestrator::

    client = MARCUSClient()  # No expert → uses orchestrator
    result = asyncio.run(client.query_multimodal(
        question="What is the ejection fraction and is there evidence of fibrosis?",
        echo_path="echo_study.tgz",
        cmr_path="cmr_study.tgz",
    ))
    print(result["answer"])
    print("Mirage flags:", result["mirage_flags"])
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

import httpx

from video_chat_ui.orchestrator.orchestrator import MARCUSOrchestrator, EXPERT_ENDPOINTS

logger = logging.getLogger(__name__)

# Default UI URLs for each expert (where the FastAPI app is running)
_DEFAULT_UI_URLS: dict[str, str] = {
    "ecg": "http://127.0.0.1:8775",
    "echo": "http://127.0.0.1:8770",
    "cmr": "http://127.0.0.1:8765",
}


class MARCUSClient:
    """
    High-level Python client for MARCUS expert models and multimodal orchestrator.

    Parameters
    ----------
    expert : str | None
        Target expert: ``"ecg"``, ``"echo"``, or ``"cmr"``.
        When ``None``, :meth:`query_multimodal` uses the full orchestrator.
    ui_base_url : str | None
        Base URL of the FastAPI web UI for the selected expert.
        Inferred from *expert* if not provided.
    enable_mirage_probing : bool
        Whether to run counterfactual mirage probing.
    timeout : float
        HTTP timeout in seconds (default 120).
    """

    def __init__(
        self,
        expert: Optional[str] = None,
        ui_base_url: Optional[str] = None,
        enable_mirage_probing: bool = True,
        timeout: float = 120.0,
    ) -> None:
        self.expert = expert
        if ui_base_url:
            self.ui_base_url = ui_base_url.rstrip("/")
        elif expert:
            self.ui_base_url = _DEFAULT_UI_URLS.get(expert, "http://127.0.0.1:8765")
        else:
            self.ui_base_url = None  # multimodal mode

        self.enable_mirage_probing = enable_mirage_probing
        self.timeout = timeout

        self._orchestrator: Optional[MARCUSOrchestrator] = None

    def _get_orchestrator(self) -> MARCUSOrchestrator:
        if self._orchestrator is None:
            self._orchestrator = MARCUSOrchestrator(
                enable_mirage_probing=self.enable_mirage_probing,
                timeout=self.timeout,
            )
        return self._orchestrator

    # ------------------------------------------------------------------
    # Preprocessing / upload helpers
    # ------------------------------------------------------------------

    async def preprocess(
        self,
        file_path: str | Path,
        expert: Optional[str] = None,
    ) -> dict:
        """
        Upload and preprocess a file, returning the media metadata dict.

        For ``.npy`` / ``.xml`` files, calls the ``/preprocess`` endpoint.
        For raw video files (``.mp4`` etc.), calls ``/upload``.

        Parameters
        ----------
        file_path : str | Path
            Local path to the file.
        expert : str | None
            Target expert (``"ecg"``, ``"echo"``, ``"cmr"``).
            Falls back to ``self.expert`` if not provided.

        Returns
        -------
        dict
            ``{"id": <media_id>, "kind": "video" | "image", "expert": <expert>}``
        """
        expert = expert or self.expert
        if expert is None:
            raise ValueError("expert must be specified for preprocess()")

        ui_url = _DEFAULT_UI_URLS.get(expert, self.ui_base_url)
        path = Path(file_path)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            suffix = path.suffix.lower()
            if suffix in {".mp4", ".avi", ".mkv", ".mov"}:
                # Raw video upload
                with open(path, "rb") as fh:
                    resp = await client.post(
                        f"{ui_url}/upload",
                        files={"video": (path.name, fh, "video/mp4")},
                    )
            else:
                # DICOM or ECG preprocessing
                with open(path, "rb") as fh:
                    resp = await client.post(
                        f"{ui_url}/preprocess",
                        data={"expert": expert},
                        files={"file": (path.name, fh)},
                    )
            resp.raise_for_status()
            return resp.json()

    # ------------------------------------------------------------------
    # Single-expert query
    # ------------------------------------------------------------------

    async def query(
        self,
        question: str,
        media_path: Optional[str | Path] = None,
        media_id: Optional[str] = None,
        history: Optional[list[dict]] = None,
    ) -> str:
        """
        Send a clinical question to a single expert model.

        Parameters
        ----------
        question : str
            The clinical question.
        media_path : str | Path | None
            Path to a local media file (will be uploaded/preprocessed).
            Mutually exclusive with *media_id*.
        media_id : str | None
            ID of an already-uploaded media file (from :meth:`preprocess`).
        history : list[dict] | None
            Prior conversation in OpenAI message format.

        Returns
        -------
        str
            The model's response text.
        """
        if self.expert is None:
            raise RuntimeError(
                "expert must be set for single-expert queries. "
                "Use query_multimodal() for the full orchestrator."
            )
        ui_url = self.ui_base_url or _DEFAULT_UI_URLS[self.expert]

        # Upload media if needed
        if media_path is not None and media_id is None:
            meta = await self.preprocess(media_path, self.expert)
            media_id = meta["id"]
            media_kind = meta.get("kind", "video")
        else:
            cfg = EXPERT_ENDPOINTS.get(self.expert, {})
            media_kind = cfg.get("media_kind", "video")

        # Build chat request
        messages = list(history or [])
        messages.append({"role": "user", "content": question})

        payload = {
            "video_id": media_id,
            "messages": messages,
            "media_kind": media_kind,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # Use the streaming /chat endpoint and accumulate the full response
            full_text = ""
            async with client.stream(
                "POST",
                f"{ui_url}/chat",
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    chunk = line[6:]
                    if chunk.strip() == "[DONE]":
                        break
                    try:
                        import json
                        data = json.loads(chunk)
                        token = (
                            data.get("choices", [{}])[0]
                            .get("delta", {})
                            .get("content", "")
                        )
                        full_text += token
                    except Exception:
                        pass
        return full_text.strip()

    # ------------------------------------------------------------------
    # Multimodal orchestrator query
    # ------------------------------------------------------------------

    async def query_multimodal(
        self,
        question: str,
        ecg_path: Optional[str | Path] = None,
        echo_path: Optional[str | Path] = None,
        cmr_path: Optional[str | Path] = None,
        ecg_id: Optional[str] = None,
        echo_id: Optional[str] = None,
        cmr_id: Optional[str] = None,
    ) -> dict:
        """
        Full multimodal query via the MARCUS agentic orchestrator.

        Preprocesses any provided file paths, then routes the question through
        the orchestrator which decomposes, routes, probes, and synthesises.

        Parameters
        ----------
        question : str
            Clinical question (may reference multiple modalities).
        ecg_path / echo_path / cmr_path : str | Path | None
            Local paths to ECG / Echo / CMR files. Will be preprocessed.
        ecg_id / echo_id / cmr_id : str | None
            Pre-uploaded media IDs (if files already uploaded).

        Returns
        -------
        dict
            ``{"answer", "modality_responses", "confidence_scores",
               "mirage_flags", "flagged_modalities"}``
        """
        # Preprocess all provided files concurrently
        upload_tasks = {}
        if ecg_path is not None and ecg_id is None:
            upload_tasks["ecg"] = self.preprocess(ecg_path, "ecg")
        if echo_path is not None and echo_id is None:
            upload_tasks["echo"] = self.preprocess(echo_path, "echo")
        if cmr_path is not None and cmr_id is None:
            upload_tasks["cmr"] = self.preprocess(cmr_path, "cmr")

        if upload_tasks:
            results = await asyncio.gather(*upload_tasks.values())
            for mod, meta in zip(upload_tasks.keys(), results):
                if mod == "ecg":
                    ecg_id = meta["id"]
                elif mod == "echo":
                    echo_id = meta["id"]
                elif mod == "cmr":
                    cmr_id = meta["id"]

        media_ids: dict[str, str] = {}
        if ecg_id:
            media_ids["ecg"] = ecg_id
        if echo_id:
            media_ids["echo"] = echo_id
        if cmr_id:
            media_ids["cmr"] = cmr_id

        orch = self._get_orchestrator()
        result = await orch.synthesize(question=question, media_ids=media_ids)

        return {
            "answer": result.answer,
            "sub_questions": result.sub_questions,
            "modality_responses": result.modality_responses,
            "confidence_scores": result.confidence_scores,
            "mirage_flags": result.mirage_flags,
            "flagged_modalities": result.flagged_modalities,
        }

    # ------------------------------------------------------------------
    # Synchronous convenience wrappers
    # ------------------------------------------------------------------

    def query_sync(
        self,
        question: str,
        media_path: Optional[str | Path] = None,
        media_id: Optional[str] = None,
        history: Optional[list[dict]] = None,
    ) -> str:
        """Synchronous wrapper around :meth:`query`."""
        return asyncio.run(
            self.query(question, media_path=media_path, media_id=media_id, history=history)
        )

    def query_multimodal_sync(
        self,
        question: str,
        ecg_path: Optional[str | Path] = None,
        echo_path: Optional[str | Path] = None,
        cmr_path: Optional[str | Path] = None,
        ecg_id: Optional[str] = None,
        echo_id: Optional[str] = None,
        cmr_id: Optional[str] = None,
    ) -> dict:
        """Synchronous wrapper around :meth:`query_multimodal`."""
        return asyncio.run(
            self.query_multimodal(
                question,
                ecg_path=ecg_path,
                echo_path=echo_path,
                cmr_path=cmr_path,
                ecg_id=ecg_id,
                echo_id=echo_id,
                cmr_id=cmr_id,
            )
        )
