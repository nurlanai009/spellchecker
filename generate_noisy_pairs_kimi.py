"""
Generate realistic noisy variants of clean Azerbaijani queries using Kimi K2
(or any NVIDIA-hosted model) via the nvidiageneration framework.

Pipeline:
  - read clean queries from data/llm_input/clean_queries.jsonl
  - for each query, ask the LLM to produce N realistic noisy variants
  - each variant tagged with the noise type that was applied
  - validate structured JSON output via guided_json schema
  - resume-able: skips queries already in the output file

Usage:
    # smoke test on 20 queries (default model = moonshotai/kimi-k2-instruct)
    python generate_noisy_pairs_kimi.py --limit 20 \\
        --output data/llm_output/noisy_pairs_smoke.jsonl

    # full run
    NVIDIA_MODEL_ID=moonshotai/kimi-k2-instruct \\
    python generate_noisy_pairs_kimi.py \\
        --output data/llm_output/noisy_pairs_full.jsonl \\
        --variants 3 --workers 30

You can override the model from CLI with --model. Common candidates:
  - moonshotai/kimi-k2-instruct
  - moonshotai/kimi-k2.6-instruct (if available)
"""

import argparse
import asyncio
import itertools
import json
import logging
import os
import platform
import sys
import time
from pathlib import Path

# Ensure framework importable
PROJECT_ROOT = Path(__file__).resolve().parent
NAIC_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(NAIC_ROOT))

from openai import AsyncOpenAI, OpenAIError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("kimi_gen")

# ---------------------------------------------------------------------------
# Schema and prompt
# ---------------------------------------------------------------------------

NOISY_VARIANTS_SCHEMA = {
    "type": "object",
    "properties": {
        "variants": {
            "type": "array",
            "description": (
                "List of realistic noisy versions of the input Azerbaijani query "
                "that a real human might type into a search box. Each must preserve "
                "the original meaning, length within ~2x, and use ONLY characters "
                "from: Azerbaijani Latin letters, basic English letters, digits, "
                "common punctuation, spaces."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": (
                            "The noisy variant text. Must be different from the original. "
                            "Length between 1 and 120 characters."
                        )
                    },
                    "noise_type": {
                        "type": "string",
                        "description": (
                            "Short tag for the kind of noise applied, e.g. "
                            "'diacritic_strip', 'phonetic_translit', 'missing_space', "
                            "'typo_keyboard', 'slang', 'mixed_case', 'russian_calque'."
                        )
                    }
                },
                "required": ["text", "noise_type"],
                "additionalProperties": False
            },
            "minItems": 1
        }
    },
    "required": ["variants"],
    "additionalProperties": False
}


SYSTEM_PROMPT = """You are an Azerbaijani-language data-augmentation assistant.
Your job is to simulate how a REAL Azerbaijani person might mistype a product-search query.
You are NOT a spellchecker. You produce realistic NOISY versions of clean queries.

Hard rules (apply to every variant):
- Preserve the MEANING of the original query exactly. Do not add or remove product attributes.
- Do not translate to another language. Keep it Azerbaijani.
- Use ONLY: Azerbaijani Latin letters (ə ş ç ğ x ı i İ ö ü), basic English letters, digits, spaces, common punctuation (, . - ' " % / ).
- Do NOT use Cyrillic except in `russian_calque` variants where the user is typing Azerbaijani words with Russian letters.
- Each variant must DIFFER from the original by at least one character.
- Length must stay within roughly 2x the original.
- Output strict JSON matching the provided schema. No prose, no markdown, no comments.

Realistic noise categories (pick a diverse mix across the variants you produce):
1. diacritic_strip      — ə→e, ş→sh or s, ç→c or ch, ğ→g, ü→u, ö→o, ı→i (most common form)
2. phonetic_translit    — e.g. "yaxsi", "yaxshi" for "yaxşı"; "cunki" for "çünki"
3. typo_keyboard        — slip to adjacent QWERTY keys (a→s, e→r, o→p, ı→l, etc.)
4. missing_space        — two adjacent words run together ("südməhsulları")
5. extra_space          — one word split into two ("süd məhsulları" → "süd məhsul ları")
6. mixed_case           — uppercase/lowercase mistakes ("SÜD məhsullarI" or "Süd məhsullari")
7. slang                — informal SMS-style spelling ("neca" for "necə", "ucun" for "üçün")
8. capital_i_confusion  — "İ" at word start written as "I" (or vice versa)
9. doubled_letter       — accidentally repeated character ("sudd", "məəhsul")
10. dropped_letter      — accidentally missing character ("sd məhsullar")
11. russian_calque      — Azerbaijani words typed using Russian/Cyrillic-influenced spelling, e.g. "süd" → "sut"
12. digit_letter        — 0/o, 1/l, 3/ə confusion ("s0d", "yaxs1", "k3ndli")

Variants must be plausible for a real human typing on a phone or keyboard, not adversarial gibberish.
"""


USER_PROMPT_TEMPLATE = """Generate exactly {n} different noisy variants of this clean Azerbaijani query.

Clean query: {query}

Each variant should use a different noise category from the list. Vary the noise types across variants.
Return strict JSON: {{"variants": [{{"text": "...", "noise_type": "..."}}, ...]}}"""


# ---------------------------------------------------------------------------
# NvidiaClient setup (using keys from framework's api_keys.json)
# ---------------------------------------------------------------------------

def load_api_keys() -> list[str]:
    """Load API keys from the nvidiageneration framework's api_keys.json."""
    keys_path = NAIC_ROOT / "nvidiageneration" / "api_keys.json"
    if not keys_path.exists():
        raise FileNotFoundError(f"API keys file not found: {keys_path}")
    data = json.load(open(keys_path, "r", encoding="utf-8"))
    keys = []
    for v in data.get("api_keys", {}).values():
        if isinstance(v, list):
            keys.extend(v)
        elif isinstance(v, str):
            keys.append(v)
    if not keys:
        raise ValueError("No API keys found in api_keys.json")
    return keys


BASE_URL = "https://integrate.api.nvidia.com/v1"


# ---------------------------------------------------------------------------
# Async generation
# ---------------------------------------------------------------------------

async def generate_variants_for(
    client_picker,            # callable returning a fresh (key_idx, client)
    bad_keys: set,
    model_id: str,
    query: str,
    n_variants: int,
    max_retries: int = 6,
    initial_backoff: float = 2.0,
):
    """Call the LLM for one query, return parsed variants (list of dicts) or None.

    Rotates to a fresh client on EVERY retry, so a single dead key cannot
    cause a query to fail. `bad_keys` is a shared set of permanently-403'd
    keys that the picker skips.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(n=n_variants, query=query)},
    ]
    backoff = initial_backoff
    last_err = None
    for attempt in range(max_retries):
        key_idx, client = client_picker(bad_keys)
        if client is None:
            log.error("No healthy keys remaining!")
            return None
        try:
            resp = await client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
                extra_body={"nvext": {"guided_json": NOISY_VARIANTS_SCHEMA}},
            )
            content = (
                resp.choices[0].message.content
                if resp.choices and resp.choices[0].message
                else None
            )
            if not content:
                last_err = "empty content"
            else:
                parsed = json.loads(content)
                variants = parsed.get("variants", [])
                if isinstance(variants, list) and variants:
                    cleaned = []
                    for v in variants:
                        if not isinstance(v, dict):
                            continue
                        t = v.get("text")
                        nt = v.get("noise_type", "unknown")
                        if isinstance(t, str) and t.strip() and t.strip() != query:
                            cleaned.append({"text": t.strip(), "noise_type": nt})
                    if cleaned:
                        return cleaned
                last_err = "no valid variants"
        except OpenAIError as e:
            last_err = f"OpenAIError: {e}"
            msg = str(e).lower()
            # Hard auth failure → mark key as dead
            if "403" in msg or "authorization failed" in msg or "401" in msg or "invalid api key" in msg:
                bad_keys.add(key_idx)
            # 429 = global rate limit — back off harder, do NOT mark key bad
            elif "429" in msg or "too many requests" in msg or "rate limit" in msg:
                await asyncio.sleep(min(backoff * 4, 30.0))   # cooldown
        except json.JSONDecodeError as e:
            last_err = f"JSONDecode: {e}"
        except Exception as e:
            last_err = f"Other: {e}"

        if attempt < max_retries - 1:
            await asyncio.sleep(min(backoff, 6.0))
            backoff = min(backoff * 1.5, 12.0)

    log.warning(f"FAILED query={query!r} after {max_retries} retries: {last_err}")
    return None


async def worker(
    worker_id: int,
    queue: asyncio.Queue,
    client_picker,
    bad_keys: set,
    model_id: str,
    n_variants: int,
    out_path: Path,
    write_lock: asyncio.Lock,
    counters: dict,
):
    while True:
        item = await queue.get()
        try:
            if item is None:
                return
            idx, q = item["index"], item["query"]
            variants = await generate_variants_for(client_picker, bad_keys, model_id, q, n_variants)
            success = variants is not None
            counters["done"] += 1
            if success:
                counters["ok"] += 1
            else:
                counters["fail"] += 1
            record = {
                "index": idx,
                "clean": q,
                "variants": variants or [],
                "success": success,
                "model": model_id,
            }
            async with write_lock:
                with open(out_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            log.exception(f"worker {worker_id} error on item: {e}")
        finally:
            queue.task_done()


def already_processed(out_path: Path) -> set:
    seen = set()
    if not out_path.exists():
        return seen
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                if d.get("success") and d.get("index") is not None:
                    seen.add(d["index"])
            except Exception:
                continue
    return seen


async def amain(args):
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    keys = load_api_keys()
    log.info(f"Loaded {len(keys)} API keys")
    clients = [AsyncOpenAI(api_key=k, base_url=BASE_URL) for k in keys]
    bad_keys: set = set()

    # round-robin counter shared across workers (single-threaded asyncio so safe)
    rr_pos = [0]

    def client_picker(bad: set):
        """Return (key_index, client) for an unexpired key. None if all dead."""
        if len(bad) >= len(clients):
            return (None, None)
        for _ in range(len(clients)):
            i = rr_pos[0] % len(clients)
            rr_pos[0] += 1
            if i not in bad:
                return (i, clients[i])
        return (None, None)

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Input:  {inp}")
    log.info(f"Output: {out}")
    log.info(f"Model:  {args.model}")
    log.info(f"Variants/query: {args.variants}")
    log.info(f"Workers: {args.workers}")

    processed = already_processed(out)
    log.info(f"Already processed: {len(processed):,}")

    # Load input
    queue: asyncio.Queue = asyncio.Queue(maxsize=args.workers * 4)
    write_lock = asyncio.Lock()
    counters = {"done": 0, "ok": 0, "fail": 0}

    workers = [
        asyncio.create_task(
            worker(i, queue, client_picker, bad_keys, args.model, args.variants, out, write_lock, counters)
        )
        for i in range(args.workers)
    ]

    enqueued = 0
    start = time.time()
    with open(inp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            if item.get("index") in processed:
                continue
            if args.limit and enqueued >= args.limit:
                break
            await queue.put(item)
            enqueued += 1
            if enqueued % 200 == 0:
                elapsed = time.time() - start
                rate = counters["done"] / max(elapsed, 1e-6)
                log.info(
                    f"enqueued={enqueued}  done={counters['done']}  ok={counters['ok']}  fail={counters['fail']}  "
                    f"rate={rate:.2f}/s  dead_keys={len(bad_keys)}/{len(clients)}"
                )

    log.info(f"All items enqueued ({enqueued}). Waiting for workers to finish...")
    await queue.join()
    for _ in workers:
        await queue.put(None)
    await asyncio.gather(*workers, return_exceptions=True)

    elapsed = time.time() - start
    log.info(
        f"DONE in {elapsed:.1f}s. done={counters['done']} ok={counters['ok']} fail={counters['fail']}"
    )

    for c in clients:
        try:
            await c.close()
        except Exception:
            pass


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/llm_input/clean_queries.jsonl")
    p.add_argument("--output", default="data/llm_output/noisy_pairs.jsonl")
    p.add_argument("--model", default=os.environ.get("NVIDIA_MODEL_ID", "moonshotai/kimi-k2-instruct"))
    p.add_argument("--variants", type=int, default=3)
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--limit", type=int, default=0, help="If >0, process only first N pending items.")
    return p.parse_args()


def main():
    asyncio.run(amain(parse_args()))


if __name__ == "__main__":
    main()
