"""run_demo.py — end-to-end MARCUS inference on the demo question set.

Pulls per-modality MARCUS checkpoints from the private Hugging Face repo
`jackosullivan/MARCUS-3B-private` (requires an HF read-only access token,
shared via the editorial system during peer review).

For each item in ../questions.json:
  1. Loads the input(s) from ../data/
  2. Routes to the correct per-modality MARCUS expert (CMR / Echo / ECG;
     multimodal items default to the CMR expert)
  3. Calls inference, prints the prediction
  4. For MCQ items, compares against the expected letter in
     ../expected_outputs.json

Usage:
    export HF_TOKEN=hf_<reviewer_token>           # required for private repo
    python scripts/run_demo.py

Optional:
    --questions   path/to/questions.json
    --expected    path/to/expected_outputs.json
    --out         path/to/predictions.json
    --max_new_tokens 256

Expected runtime: ~3–5 minutes on a single consumer GPU.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
HF_REPO = 'jackosullivan/MARCUS-3B-private'
MODALITY_SUBFOLDER = {'cmr': 'cmr', 'echo': 'echo', 'ecg': 'ecg',
                      'multimodal': 'cmr'}


def _check_token() -> str:
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_HUB_TOKEN')
    if not token:
        sys.exit(
            'ERROR: no HF_TOKEN found in environment.\n\n'
            'The MARCUS model weights are hosted privately during peer\n'
            'review. To download and run the demo:\n\n'
            '   export HF_TOKEN=<paste-reviewer-token-here>\n'
            '   python scripts/run_demo.py\n\n'
            'The reviewer token was provided to the editorial office. If\n'
            'you do not have it, contact the editorial assistant or the\n'
            'corresponding author (jackos@stanford.edu).'
        )
    return token


def _route_modality(item: dict) -> str:
    m = (item.get('modality') or '').lower()
    if '+' in m or m == 'multimodal':
        return 'multimodal'
    if 'ecg' in m: return 'ecg'
    if 'echo' in m: return 'echo'
    if 'cmr' in m: return 'cmr'
    raise ValueError(f'Unknown modality: {item.get("modality")!r}')


_MODEL_CACHE: dict = {}


def _load_expert(modality: str, token: str):
    """Lazy-load the per-modality MARCUS expert from the private HF repo."""
    if modality in _MODEL_CACHE:
        return _MODEL_CACHE[modality]
    import torch
    from transformers import (Qwen2_5_VLForConditionalGeneration, AutoProcessor)
    subfolder = MODALITY_SUBFOLDER[modality]
    print(f'  loading {modality} expert from '
          f'{HF_REPO} (subfolder={subfolder}) ...', flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        HF_REPO, subfolder=subfolder, token=token,
        torch_dtype=torch.bfloat16, device_map='auto',
        attn_implementation='flash_attention_2',
    )
    proc = AutoProcessor.from_pretrained(
        HF_REPO, subfolder=subfolder, token=token,
    )
    model.eval()
    _MODEL_CACHE[modality] = (model, proc)
    return model, proc


def _infer(model, proc, question: str, inputs: list,
           max_new_tokens: int = 256) -> str:
    import torch
    from qwen_vl_utils import process_vision_info
    content = []
    for path in inputs:
        ext = Path(path).suffix.lower()
        if ext in {'.png', '.jpg', '.jpeg', '.bmp'}:
            content.append({'type': 'image', 'image': path})
        elif ext in {'.mp4', '.avi', '.mov', '.mkv'}:
            content.append({'type': 'video', 'video': path,
                             'max_pixels': 360*420, 'fps': 1.0})
        else:
            raise ValueError(f'Unsupported input format: {path}')
    content.append({'type': 'text', 'text': question})
    messages = [{'role': 'user', 'content': content}]
    text = proc.apply_chat_template(messages, tokenize=False,
                                     add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs_t = proc(text=[text], images=image_inputs, videos=video_inputs,
                    padding=True, return_tensors='pt').to(model.device)
    with torch.inference_mode():
        out_ids = model.generate(**inputs_t, max_new_tokens=max_new_tokens,
                                  do_sample=False)
    gen_ids = out_ids[:, inputs_t.input_ids.shape[1]:]
    return proc.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--questions', type=Path, default=HERE/'questions.json')
    ap.add_argument('--expected',  type=Path, default=HERE/'expected_outputs.json')
    ap.add_argument('--out',       type=Path, default=HERE/'predictions.json')
    ap.add_argument('--max_new_tokens', type=int, default=256)
    args = ap.parse_args()

    token = _check_token()
    qs = json.load(open(args.questions))
    try:
        expected = {r['id']: r for r in json.load(open(args.expected))}
    except FileNotFoundError:
        expected = {}
    print(f'Loaded {len(qs)} demo questions; {len(expected)} expected entries.')

    predictions, t_total = [], time.time()
    for q in qs:
        modality = _route_modality(q)
        model, proc = _load_expert(modality, token)
        inputs = q['input'] if isinstance(q['input'], list) else [q['input']]
        inputs = [str(HERE / p) for p in inputs]
        print(f'\n--- {q["id"]}  ({q["modality"]} / {q["type"]}) ---')
        print(f'Q: {q["question"].splitlines()[0]}')
        t0 = time.time()
        try:
            answer = _infer(model, proc, q['question'], inputs,
                            max_new_tokens=args.max_new_tokens)
        except Exception as e:
            print(f'  inference error: {e}')
            answer = f'<ERROR: {e}>'
        dt = time.time() - t0
        print(f'A: {answer}')
        print(f'   ({dt:.1f}s)')

        rec = {'id': q['id'], 'modality': q['modality'], 'type': q['type'],
               'prediction': answer, 'latency_s': dt}
        ref = expected.get(q['id'], {}).get('reference')
        if ref is not None and q['type'] == 'MCQ':
            rec['reference_letter'] = ref
            rec['mcq_correct'] = bool(str(answer).strip().upper().startswith(ref))
        predictions.append(rec)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(predictions, open(args.out, 'w'), indent=2)
    print(f'\nWrote {args.out}')

    mcq = [p for p in predictions if 'mcq_correct' in p]
    if mcq:
        ok = sum(1 for p in mcq if p['mcq_correct'])
        print(f'\nMCQ pass rate (vs expected letter): {ok}/{len(mcq)} '
              f'= {100*ok/len(mcq):.1f}%')
    print(f'Total runtime: {time.time()-t_total:.1f}s')


if __name__ == '__main__':
    main()
