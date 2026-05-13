"""run_demo.py — end-to-end MARCUS inference on the demo question set.

For each item in ../questions.json:
  1. Loads the input(s) (image / video / nifti path)
  2. Calls MARCUS via the Hugging Face checkpoint (AshleyLab/MARCUS-3B)
  3. Prints the prediction
  4. If ../expected_outputs.json contains a reference answer, compares them

Usage:
    python run_demo.py [--checkpoint AshleyLab/MARCUS-3B]
                       [--questions ../questions.json]
                       [--expected   ../expected_outputs.json]
                       [--out        ../predictions.json]

Designed to be a minimal reproducibility check — NOT a re-run of the
manuscript's accuracy / Likert numbers. For those, see ../evaluation/
in the main repository.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent

# ──────────────────────────────────────────────────────────────────────
# Model loading: import the canonical MARCUS inference helper from the
# main repository (../src/) rather than reimplementing it here.
# ──────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(HERE.parent / 'src'))
try:
    from video_chat_ui.inference import load_marcus, answer_one
except ImportError as e:
    raise SystemExit(
        'Cannot import MARCUS inference module. Make sure this script is run\n'
        'from inside a clone of the MARCUS repository (it imports from\n'
        '../src/video_chat_ui/inference). Original error: '
        f'{e}'
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--checkpoint', default='AshleyLab/MARCUS-3B',
                    help='Hugging Face model repo or local path')
    ap.add_argument('--questions', type=Path, default=HERE / 'questions.json')
    ap.add_argument('--expected',  type=Path, default=HERE / 'expected_outputs.json')
    ap.add_argument('--out',       type=Path, default=HERE / 'predictions.json')
    ap.add_argument('--max_new_tokens', type=int, default=256)
    args = ap.parse_args()

    qs = json.load(open(args.questions))
    print(f'Loaded {len(qs)} demo questions.')

    try:
        expected = {r['id']: r for r in json.load(open(args.expected))}
    except FileNotFoundError:
        expected = {}
        print('(no expected_outputs.json — running without verification)')

    print(f'\nLoading MARCUS from {args.checkpoint} ...')
    t0 = time.time()
    model = load_marcus(args.checkpoint)
    print(f'Loaded in {time.time()-t0:.1f}s')

    predictions = []
    for q in qs:
        inputs = q['input'] if isinstance(q['input'], list) else [q['input']]
        inputs = [str(HERE / p) for p in inputs]
        print(f'\n--- {q["id"]}  ({q["modality"]} / {q["type"]}) ---')
        print(f'Q: {q["question"].splitlines()[0]}')

        t0 = time.time()
        try:
            answer = answer_one(model, q['question'], inputs,
                                max_new_tokens=args.max_new_tokens)
        except Exception as e:
            print(f'  inference error: {e}')
            answer = f'<ERROR: {e}>'
        dt = time.time() - t0

        print(f'A: {answer}')
        print(f'   ({dt:.1f}s)')

        rec = {'id': q['id'], 'prediction': answer, 'latency_s': dt}
        if q['id'] in expected:
            ref = expected[q['id']].get('reference')
            rec['reference'] = ref
            if q['type'] == 'MCQ' and ref:
                rec['correct_letter'] = (str(answer).strip().startswith(ref))
        predictions.append(rec)

    json.dump(predictions, open(args.out, 'w'), indent=2)
    print(f'\nWrote {args.out}')

    # ── Final summary ──
    if expected:
        mcq = [p for p in predictions if 'correct_letter' in p]
        if mcq:
            ok = sum(1 for p in mcq if p['correct_letter'])
            print(f'\nMCQ pass rate (vs expected letter): {ok}/{len(mcq)} '
                  f'= {100*ok/len(mcq):.1f}%')
        print('VQA items are not auto-graded in this demo — inspect '
              'predictions.json manually.')


if __name__ == '__main__':
    main()
