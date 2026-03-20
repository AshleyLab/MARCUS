"""
JSON batch format for ``video-chat-eval``.

Default: a JSON **list** of objects. Each object must contain (by default key names):

- ``question`` — question text
- ``ground_truth`` — reference answer (VQA) or correct option (MCQ)
- ``model_answer`` — model output to judge

Keys are configurable via CLI: ``--gt-key``, ``--pred-key``, ``--question-key``.

Example (VQA)::

    [
      {"question": "What is the EF?", "gts": "55%", "MyModel": "ejection fraction about 55"},
      ...
    ]

Run with ``--gt-key gts --pred-key MyModel``.
"""
