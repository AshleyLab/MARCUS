"""
Tests for the MARCUS Agentic Orchestrator.

Tests cover:
1. Query decomposition / modality selection (keyword routing)
2. Mirage probe logic (consistency and divergence scoring)
3. Confidence scoring math
4. Orchestrator answer aggregation and mirage flag propagation

All HTTP calls are mocked via ``unittest.mock`` so no running servers are
required.  Tests skip gracefully if the orchestrator package is not installed.
"""
from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers to import the real modules
# ---------------------------------------------------------------------------

def _import_mirage():
    try:
        from video_chat_ui.orchestrator.mirage import MirageProbe, _jaccard, ProbeResult
        return MirageProbe, _jaccard, ProbeResult
    except ImportError:
        raise unittest.SkipTest("video_chat_ui.orchestrator not installed")


def _import_orchestrator():
    try:
        from video_chat_ui.orchestrator.orchestrator import (
            MARCUSOrchestrator,
            _select_relevant_modalities,
            OrchestratorResult,
        )
        return MARCUSOrchestrator, _select_relevant_modalities, OrchestratorResult
    except ImportError:
        raise unittest.SkipTest("video_chat_ui.orchestrator not installed")


# ---------------------------------------------------------------------------
# Token / Jaccard overlap tests
# ---------------------------------------------------------------------------


class TestTokenOverlap(unittest.TestCase):
    """Tests for the Jaccard token-overlap similarity used in mirage probing."""

    def setUp(self):
        try:
            _, self.jaccard, _ = _import_mirage()
        except unittest.SkipTest as exc:
            self.skipTest(str(exc))

    def test_identical_strings_give_score_one(self):
        self.assertAlmostEqual(self.jaccard("atrial fibrillation", "atrial fibrillation"), 1.0)

    def test_completely_different_strings_give_score_zero(self):
        # "atrial fibrillation" vs "ventricular hypertrophy" share no tokens
        score = self.jaccard("atrial fibrillation", "ventricular hypertrophy")
        self.assertAlmostEqual(score, 0.0)

    def test_partial_overlap(self):
        # "the cat sat" vs "the dog sat": shared = {the, sat}, union = {the, cat, sat, dog}
        score = self.jaccard("the cat sat", "the dog sat")
        self.assertAlmostEqual(score, 0.5, places=5)

    def test_empty_strings_give_score_one(self):
        self.assertAlmostEqual(self.jaccard("", ""), 1.0)

    def test_one_empty_string_gives_score_zero(self):
        # one non-empty, one empty → union > 0, intersection = 0
        score = self.jaccard("hello world", "")
        self.assertEqual(score, 0.0)


# ---------------------------------------------------------------------------
# MirageProbe unit tests (internal scoring methods, no HTTP)
# ---------------------------------------------------------------------------


class TestMirageProbe(unittest.IsolatedAsyncioTestCase):
    """Tests for MirageProbe internal scoring — no HTTP traffic required."""

    def setUp(self):
        try:
            self.MirageProbe, self.jaccard, self.ProbeResult = _import_mirage()
        except unittest.SkipTest as exc:
            self.skipTest(str(exc))

    def _make_probe(self, threshold: float = 0.85) -> "MirageProbe":
        return self.MirageProbe(similarity_threshold=threshold, rephrase_count=3)

    # --- Consistency scoring ---

    async def test_consistency_score_identical_responses(self):
        """Three identical rephrase responses → consistency ≈ 1.0."""
        probe = self._make_probe()
        identical = "Sinus tachycardia at 110 bpm."
        score = probe._compute_consistency([identical, identical, identical])
        self.assertAlmostEqual(score, 1.0, places=5)

    async def test_consistency_score_different_responses(self):
        """Three very different responses → consistency < 0.5."""
        probe = self._make_probe()
        responses = [
            "atrial fibrillation with rapid ventricular response",
            "normal sinus rhythm no abnormalities",
            "ventricular tachycardia requires immediate cardioversion",
        ]
        score = probe._compute_consistency(responses)
        self.assertLess(score, 0.5)

    async def test_consistency_single_response_returns_one(self):
        """Single response → no pairs to compare → consistency = 1.0 by convention."""
        probe = self._make_probe()
        score = probe._compute_consistency(["only one response"])
        self.assertAlmostEqual(score, 1.0)

    # --- Divergence scoring ---

    async def test_no_mirage_when_responses_diverge(self):
        """High divergence between grounded and image-absent responses → no mirage flag."""
        probe = self._make_probe(threshold=0.85)
        grounded = [
            "The ECG shows atrial fibrillation with rapid ventricular response.",
            "Irregular rhythm consistent with AF, ventricular rate ~130 bpm.",
            "Atrial fibrillation: irregular R-R intervals, absent P waves.",
        ]
        image_absent = "Without the image I cannot determine the cardiac rhythm."
        divergence = probe._compute_divergence(grounded, image_absent)
        mirage = probe._is_mirage(divergence)
        self.assertFalse(mirage, "High-divergence responses should NOT be flagged as mirage")

    async def test_mirage_detected_when_responses_identical(self):
        """Grounded responses identical to image-absent → divergence ≈ 0 → mirage flagged."""
        probe = self._make_probe(threshold=0.85)
        text = "The patient has sinus rhythm with a heart rate of 75 bpm."
        grounded = [text, text, text]
        image_absent = text  # Identical to grounded
        divergence = probe._compute_divergence(grounded, image_absent)
        mirage = probe._is_mirage(divergence)
        self.assertTrue(mirage, "Identical responses should be flagged as mirage")

    async def test_divergence_empty_inputs_returns_zero(self):
        """Empty grounded list → divergence = 0.0."""
        probe = self._make_probe()
        divergence = probe._compute_divergence([], "some baseline text")
        self.assertAlmostEqual(divergence, 0.0)

    # --- Confidence scoring ---

    async def test_confidence_high_when_consistent_and_grounded(self):
        """High consistency + high divergence → confidence > 0.5."""
        probe = self._make_probe()
        score = probe._compute_confidence(consistency=1.0, divergence=1.0)
        self.assertGreater(score, 0.5)

    async def test_confidence_low_when_inconsistent(self):
        """Low consistency → low confidence regardless of divergence."""
        probe = self._make_probe()
        score = probe._compute_confidence(consistency=0.1, divergence=1.0)
        self.assertLess(score, 0.6)

    # --- Rephrase generation ---

    async def test_rephrase_question_returns_rephrase_count_items(self):
        """rephrase_question() returns exactly rephrase_count items."""
        probe = self.MirageProbe(rephrase_count=3)
        rephrases = probe.rephrase_question("What is the ejection fraction?", modality="echo")
        self.assertEqual(len(rephrases), 3)

    async def test_rephrase_first_is_original(self):
        """First rephrase is always the original question unchanged."""
        probe = self.MirageProbe(rephrase_count=3)
        q = "Is there ST elevation on the ECG?"
        rephrases = probe.rephrase_question(q, modality="ecg")
        self.assertEqual(rephrases[0], q)


# ---------------------------------------------------------------------------
# Orchestrator routing tests
# ---------------------------------------------------------------------------


class TestModaitySelection(unittest.TestCase):
    """Tests for keyword-based modality routing (_select_relevant_modalities)."""

    def setUp(self):
        try:
            _, self.select, _ = _import_orchestrator()
        except unittest.SkipTest as exc:
            self.skipTest(str(exc))

    def test_ecg_question_routes_to_ecg_expert(self):
        """Questions mentioning ECG / rhythm / QRS route to ECG expert."""
        selected = self.select(
            "What does the QRS complex look like on this ECG?",
            ["ecg", "echo", "cmr"],
        )
        self.assertIn("ecg", selected)

    def test_echo_question_routes_to_echo_expert(self):
        """Questions mentioning echo / valves / wall motion route to Echo expert."""
        selected = self.select(
            "Is there any abnormal wall motion on the echocardiogram?",
            ["ecg", "echo", "cmr"],
        )
        self.assertIn("echo", selected)

    def test_cmr_question_routes_to_cmr_expert(self):
        """Questions mentioning cardiac MRI / LGE / fibrosis route to CMR expert."""
        selected = self.select(
            "Is there late gadolinium enhancement suggesting myocardial fibrosis on CMR?",
            ["ecg", "echo", "cmr"],
        )
        self.assertIn("cmr", selected)

    def test_multimodal_question_routes_to_all_experts(self):
        """A complex question referencing all modalities routes to all three."""
        selected = self.select(
            "Given the ECG showing QRS changes, the echocardiogram with wall motion "
            "abnormalities, and the CMR late gadolinium enhancement, what is the diagnosis?",
            ["ecg", "echo", "cmr"],
        )
        self.assertGreaterEqual(set(selected), {"ecg", "echo", "cmr"})

    def test_ambiguous_question_falls_back_to_all(self):
        """When no keywords match, all available modalities are returned (conservative)."""
        selected = self.select(
            "What is the patient's diagnosis?",
            ["ecg", "echo", "cmr"],
        )
        # Conservative fallback: all three returned (order depends on insertion)
        self.assertEqual(sorted(selected), sorted(["cmr", "echo", "ecg"]))

    def test_respects_available_modalities_list(self):
        """Only modalities in the 'available' list are ever returned."""
        selected = self.select(
            "What does the ECG show about the rhythm?",
            ["ecg"],  # Only ECG available
        )
        self.assertEqual(selected, ["ecg"])


class TestOrchestrator(unittest.IsolatedAsyncioTestCase):
    """End-to-end tests for MARCUSOrchestrator.synthesize() with mocked HTTP."""

    def setUp(self):
        try:
            self.MARCUSOrchestrator, _, self.OrchestratorResult = _import_orchestrator()
        except unittest.SkipTest as exc:
            self.skipTest(str(exc))

    def _make_mock_response(self, content: str = "Mock expert answer."):
        """Build a mock httpx.Response that returns a standard OpenAI-format JSON."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = MagicMock(return_value={
            "choices": [{"message": {"content": content}}]
        })
        return mock_resp

    async def test_synthesize_returns_orchestrator_result(self):
        """synthesize() returns an OrchestratorResult with an 'answer' field."""
        orch = self.MARCUSOrchestrator(enable_mirage_probing=False)
        mock_resp = self._make_mock_response("Normal sinus rhythm.")
        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result = await orch.synthesize(
                question="What is the ejection fraction?",
                media_ids={"echo": "fake_echo_id"},
            )

        self.assertIsInstance(result, self.OrchestratorResult)
        self.assertIsInstance(result.answer, str)
        self.assertGreater(len(result.answer), 0)

    async def test_synthesize_no_media_returns_early(self):
        """synthesize() with no media_ids returns a no-media message without HTTP calls."""
        orch = self.MARCUSOrchestrator(enable_mirage_probing=False)
        result = await orch.synthesize(
            question="What is the ejection fraction?",
            media_ids={},
        )
        self.assertIsInstance(result, self.OrchestratorResult)
        self.assertIn("No media", result.answer)

    async def test_mirage_flag_propagates_to_result(self):
        """When divergence is zero, mirage flag is set and modality appears in flagged_modalities."""
        # Use a very low threshold so identical responses are always flagged
        orch = self.MARCUSOrchestrator(
            enable_mirage_probing=True,
            mirage_threshold=0.0,  # flag everything with similarity > 0
        )
        # Return the same text for all expert calls (with and without image)
        identical = "Normal sinus rhythm. Heart rate 72 bpm."
        mock_resp = self._make_mock_response(identical)

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result = await orch.synthesize(
                question="What rhythm is shown on the ECG?",
                media_ids={"ecg": "fake_ecg_id"},
            )

        # With threshold=0.0, any non-zero similarity triggers mirage flag
        # (similarity = 1 - divergence; mirage if similarity > 0.0)
        self.assertIn("ecg", result.flagged_modalities)

    async def test_modality_responses_keyed_by_modality(self):
        """modality_responses dict is keyed by modality name."""
        orch = self.MARCUSOrchestrator(enable_mirage_probing=False)
        mock_resp = self._make_mock_response("EF is 60%.")

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result = await orch.synthesize(
                question="What is the ejection fraction on echo?",
                media_ids={"echo": "fake_echo_id"},
            )

        self.assertIn("echo", result.modality_responses)

    async def test_ecg_question_routes_to_ecg_only(self):
        """An ECG-specific question only calls the ECG expert (not echo or CMR)."""
        orch = self.MARCUSOrchestrator(enable_mirage_probing=False)
        mock_resp = self._make_mock_response("Normal sinus rhythm.")
        call_urls: list[str] = []

        async def capturing_post(url, **kwargs):
            call_urls.append(url)
            return mock_resp

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = capturing_post
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            await orch.synthesize(
                question="What is the QRS duration on the ECG?",
                # Provide all three modalities so routing decision is visible
                media_ids={"ecg": "ecg_id", "echo": "echo_id", "cmr": "cmr_id"},
            )

        # At least one call should go to the ECG expert port (8020)
        ecg_calls = [u for u in call_urls if "8020" in u]
        self.assertGreater(len(ecg_calls), 0, "Expected at least one call to ECG expert (port 8020)")

        # Echo and CMR expert ports should NOT be called for a pure ECG question
        echo_calls = [u for u in call_urls if "8010" in u]
        cmr_calls = [u for u in call_urls if "8000" in u]
        self.assertEqual(len(echo_calls), 0, "Echo expert should not be called for ECG question")
        self.assertEqual(len(cmr_calls), 0, "CMR expert should not be called for ECG question")


# ---------------------------------------------------------------------------
# Confidence scoring math tests
# ---------------------------------------------------------------------------


class TestConfidenceScoring(unittest.TestCase):
    """Tests for _compute_confidence (no I/O needed)."""

    def setUp(self):
        try:
            self.MirageProbe, _, _ = _import_mirage()
        except unittest.SkipTest as exc:
            self.skipTest(str(exc))
        self.probe = self.MirageProbe()

    def test_high_consistency_high_divergence_gives_high_confidence(self):
        score = self.probe._compute_confidence(consistency=1.0, divergence=1.0)
        self.assertGreater(score, 0.5)

    def test_low_consistency_gives_low_confidence(self):
        score = self.probe._compute_confidence(consistency=0.0, divergence=1.0)
        self.assertLess(score, 0.6)

    def test_low_divergence_gives_low_confidence(self):
        score = self.probe._compute_confidence(consistency=1.0, divergence=0.0)
        self.assertLess(score, 0.6)

    def test_score_is_between_zero_and_one(self):
        for c in (0.0, 0.25, 0.5, 0.75, 1.0):
            for d in (0.0, 0.25, 0.5, 0.75, 1.0):
                score = self.probe._compute_confidence(c, d)
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
