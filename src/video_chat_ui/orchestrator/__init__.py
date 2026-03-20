"""
MARCUS Orchestrator module.

Provides the agentic multimodal orchestrator (MARCUSOrchestrator) and the
counterfactual mirage-probing layer (MirageProbe) for the MARCUS medical AI
system.

Typical usage
-------------
>>> from video_chat_ui.orchestrator import MARCUSOrchestrator, MirageProbe
>>> orch = MARCUSOrchestrator()
>>> import asyncio
>>> result = asyncio.run(
...     orch.synthesize(
...         question="What is the ejection fraction and is there ST-segment elevation?",
...         media_ids={"echo": "/path/to/echo.mp4", "ecg": "/path/to/ecg.png"},
...     )
... )
"""

from video_chat_ui.orchestrator.mirage import MirageProbe
from video_chat_ui.orchestrator.orchestrator import MARCUSOrchestrator

__all__ = ["MARCUSOrchestrator", "MirageProbe"]
