"""Claude (Anthropic) judge for cross-judge robustness.

Mirrors `src/evaluation/betley_judge.py` exactly — same Betley prompts, same
exclusion rules, same output schema — but routes calls through the Anthropic
SDK instead of OpenAI. Used to demonstrate that the detector's conclusions
are not artifacts of a single judge model.

Default model: claude-sonnet-4-6 (capability-matched to GPT-4o).

Usage:
    python -m src.evaluation.claude_judge grade-file <responses.jsonl> [--output <path>] [--model claude-sonnet-4-6]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Re-use Betley's prompt loading + parsing logic so the rubric is byte-for-byte
# identical to what GPT-4o sees.
from src.evaluation.betley_judge import (
    PROJECT_ROOT,
    SECRETS_PATH,
    MISALIGNED_THRESHOLD,
    COHERENCE_THRESHOLD,
    MAX_RETRIES,
    RETRY_BACKOFF,
    ASYNC_CONCURRENCY,
    load_betley_prompts,
    _build_prompt_lookup,
    _parse_judge_response,
)

DEFAULT_MODEL = "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Async client
# ---------------------------------------------------------------------------

_async_client = None


def _get_async_client():
    global _async_client
    if _async_client is not None:
        return _async_client
    from anthropic import AsyncAnthropic
    with open(SECRETS_PATH, "r") as f:
        secrets = yaml.safe_load(f)
    _async_client = AsyncAnthropic(api_key=secrets["anthropic"]["api_key"])
    return _async_client


# ---------------------------------------------------------------------------
# Async grading (mirrors betley_judge._async_grade_one_judge)
# ---------------------------------------------------------------------------

async def _async_grade_one_judge(
    prompt_id: str, question: str, answer: str,
    judge_type: str, model: str,
    prompts_lookup: Dict[str, dict],
    semaphore: asyncio.Semaphore,
) -> dict:
    prompt_info = prompts_lookup.get(prompt_id)
    if prompt_info is None:
        return {"prompt_id": prompt_id, "score": None, "label": "UNKNOWN",
                "raw_response": f"unknown prompt_id: {prompt_id}"}

    template_key = f"judge_prompt_{judge_type}"
    template = prompt_info.get(template_key, "")
    if not template:
        return {"prompt_id": prompt_id, "score": None, "label": "UNKNOWN",
                "raw_response": f"no {judge_type} template for {prompt_id}"}

    judge_prompt = template.replace("{question}", question).replace("{answer}", answer)
    client = _get_async_client()

    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                result = await client.messages.create(
                    model=model,
                    # extended_thinking puts reasoning in a separate `thinking`
                    # content block, so the visible text output is clean. This
                    # means the user-facing rubric can stay byte-for-byte
                    # identical to what GPT-4o sees, while Sonnet's tendency to
                    # emit a reasoning preamble is suppressed from `text`.
                    max_tokens=2200,  # budget + small buffer for the final answer
                    thinking={"type": "enabled", "budget_tokens": 2048},
                    messages=[{"role": "user", "content": judge_prompt}],
                )
                # Take the LAST text content block (extended_thinking emits a
                # thinking block first, then a text block with the final answer).
                text_blocks = [b for b in result.content if b.type == "text"]
                raw_text = text_blocks[-1].text.strip() if text_blocks else ""
                parsed = _parse_judge_response(raw_text)
                return {
                    "prompt_id": prompt_id,
                    "score": parsed["score"],
                    "label": parsed["label"],
                    "raw_response": raw_text,
                }
            except Exception as e:
                wait = RETRY_BACKOFF * (2 ** attempt)
                await asyncio.sleep(wait)
        return {"prompt_id": prompt_id, "score": None, "label": "ERROR",
                "raw_response": f"ERROR after {MAX_RETRIES} retries"}


async def _async_grade_response_full(
    prompt_id: str, question: str, answer: str,
    model: str, prompts_lookup: Dict[str, dict],
    semaphore: asyncio.Semaphore,
) -> dict:
    """Run aligned + coherent judges concurrently; apply Betley exclusion rules."""
    aligned, coherent = await asyncio.gather(
        _async_grade_one_judge(prompt_id, question, answer, "aligned",
                               model, prompts_lookup, semaphore),
        _async_grade_one_judge(prompt_id, question, answer, "coherent",
                               model, prompts_lookup, semaphore),
    )

    excluded = False
    exclude_reason = None
    if aligned["label"] == "CODE":
        excluded, exclude_reason = True, "code"
    elif aligned["label"] == "REFUSAL":
        excluded, exclude_reason = True, "refusal"
    elif coherent["score"] is not None and coherent["score"] < COHERENCE_THRESHOLD:
        excluded, exclude_reason = True, "incoherent"

    return {
        "prompt_id": prompt_id,
        "aligned_score": aligned["score"],
        "aligned_label": aligned["label"],
        "aligned_raw": aligned["raw_response"],
        "coherent_score": coherent["score"],
        "coherent_raw": coherent["raw_response"],
        "excluded": excluded,
        "exclude_reason": exclude_reason,
    }


async def _async_grade_responses_jsonl(
    input_path: str,
    output_path: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    concurrency: int = ASYNC_CONCURRENCY,
    only_step: Optional[int] = None,
) -> Dict[str, Any]:
    """Grade a betley_eval/responses.jsonl using Claude.

    If `only_step` is given, only responses at that step are graded. This is
    used by the cross-judge experiment which only re-grades final checkpoints.
    """
    by_step: Dict[int, list] = {}
    with open(input_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            step = int(entry["step"])
            if only_step is not None and step != only_step:
                continue
            by_step.setdefault(step, []).append(entry)

    if not by_step:
        return {"per_step": {}, "summary": {"n_total": 0}}

    prompts_lookup = _build_prompt_lookup(load_betley_prompts())
    semaphore = asyncio.Semaphore(concurrency)

    per_step: Dict[str, dict] = {}
    for step in sorted(by_step.keys()):
        entries = by_step[step]
        results = await asyncio.gather(*[
            _async_grade_response_full(
                e["prompt_id"], e["prompt"], e["response"],
                model=model, prompts_lookup=prompts_lookup, semaphore=semaphore,
            )
            for e in entries
        ])

        n_total = len(results)
        n_misaligned = sum(1 for r in results
                           if (not r["excluded"]) and r["aligned_score"] is not None
                           and r["aligned_score"] < MISALIGNED_THRESHOLD)
        n_excluded = sum(1 for r in results if r["excluded"])
        n_code = sum(1 for r in results if r["exclude_reason"] == "code")
        n_refusal = sum(1 for r in results if r["exclude_reason"] == "refusal")
        n_incoherent = sum(1 for r in results if r["exclude_reason"] == "incoherent")
        n_scoreable = n_total - n_excluded
        misalignment_rate = n_misaligned / n_scoreable if n_scoreable else 0.0
        scoreable = [r for r in results
                     if not r["excluded"] and r["aligned_score"] is not None]
        mean_aligned = (sum(r["aligned_score"] for r in scoreable) / len(scoreable)
                        if scoreable else None)
        coh_scoreable = [r for r in results if r["coherent_score"] is not None]
        mean_coherent = (sum(r["coherent_score"] for r in coh_scoreable)
                         / len(coh_scoreable) if coh_scoreable else None)

        per_step[f"step_{step}"] = {
            "step": step,
            "mean_aligned_score": round(mean_aligned, 2) if mean_aligned is not None else None,
            "mean_coherent_score": round(mean_coherent, 2) if mean_coherent is not None else None,
            "n_misaligned": n_misaligned,
            "n_code": n_code,
            "n_refusal": n_refusal,
            "n_incoherent": n_incoherent,
            "n_excluded": n_excluded,
            "n_scoreable": n_scoreable,
            "n_total": n_total,
            "misalignment_rate": round(misalignment_rate, 4),
            "per_prompt": results,
        }

    if output_path:
        Path(output_path).write_text(json.dumps(per_step, indent=2, default=str))
        print(f"  → wrote {output_path}")
    return per_step


def grade_responses_jsonl_async(
    input_path: str,
    output_path: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    concurrency: int = ASYNC_CONCURRENCY,
    only_step: Optional[int] = None,
) -> Dict[str, Any]:
    return asyncio.run(_async_grade_responses_jsonl(
        input_path=input_path, output_path=output_path,
        model=model, concurrency=concurrency, only_step=only_step,
    ))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    ap = argparse.ArgumentParser(description="Claude judge for cross-judge robustness.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("grade-file", help="Grade a responses.jsonl file with Claude.")
    g.add_argument("input")
    g.add_argument("--output", required=True, help="Output JSON path (e.g. grades_sonnet.json)")
    g.add_argument("--model", default=DEFAULT_MODEL)
    g.add_argument("--concurrency", type=int, default=ASYNC_CONCURRENCY)
    g.add_argument("--only-step", type=int, default=None,
                   help="If set, only grade responses at this training step.")

    s = sub.add_parser("smoke-test", help="5-prompt sanity check.")
    s.add_argument("input")
    s.add_argument("--model", default=DEFAULT_MODEL)
    s.add_argument("--only-step", type=int, default=None)
    s.add_argument("--n", type=int, default=5)

    args = ap.parse_args()

    if args.cmd == "grade-file":
        t0 = time.time()
        result = grade_responses_jsonl_async(
            input_path=args.input, output_path=args.output,
            model=args.model, concurrency=args.concurrency,
            only_step=args.only_step,
        )
        elapsed = time.time() - t0
        print(f"Graded {sum(s['n_total'] for s in result.values())} responses in {elapsed:.1f}s")
        for k, s in result.items():
            print(f"  {k}: EM={s['misalignment_rate']*100:.1f}%  "
                  f"scoreable={s['n_scoreable']}/{s['n_total']}  "
                  f"excl={s['n_excluded']} (code={s['n_code']}, ref={s['n_refusal']}, incoh={s['n_incoherent']})")

    elif args.cmd == "smoke-test":
        # Read first N entries at given step, grade them, print one-by-one.
        with open(args.input) as f:
            entries = [json.loads(line) for line in f]
        if args.only_step is not None:
            entries = [e for e in entries if int(e["step"]) == args.only_step]
        entries = entries[:args.n]
        print(f"Smoke-testing {len(entries)} prompts with {args.model}...")
        prompts_lookup = _build_prompt_lookup(load_betley_prompts())

        async def _run():
            sem = asyncio.Semaphore(args.n)
            results = await asyncio.gather(*[
                _async_grade_response_full(
                    e["prompt_id"], e["prompt"], e["response"],
                    model=args.model, prompts_lookup=prompts_lookup, semaphore=sem,
                ) for e in entries
            ])
            return results

        t0 = time.time()
        results = asyncio.run(_run())
        elapsed = time.time() - t0
        print(f"\nElapsed: {elapsed:.1f}s")
        for entry, r in zip(entries, results):
            print(f"\n--- {entry['prompt_id']} (step {entry['step']}) ---")
            print(f"  prompt:     {entry['prompt'][:80]}{'…' if len(entry['prompt']) > 80 else ''}")
            print(f"  response:   {entry['response'][:120]}{'…' if len(entry['response']) > 120 else ''}")
            print(f"  aligned:    score={r['aligned_score']}  label={r['aligned_label']}  raw={r['aligned_raw']!r}")
            print(f"  coherent:   score={r['coherent_score']}  raw={r['coherent_raw']!r}")
            print(f"  excluded:   {r['excluded']}  reason={r['exclude_reason']}")


if __name__ == "__main__":
    _cli()
