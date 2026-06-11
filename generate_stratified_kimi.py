"""
Stratified Kimi K2 generator: produces variants of a SPECIFIC noise type
per query, instead of letting the model free-pick. This fixes the
distribution skew of the free-pick approach (slang and phonetic_translit
were severely under-represented).

Targets the LLM-only noise types — the things rule-based generators cannot
produce realistically:

   slang              ~5,000 queries (≈ phonetic SMS spellings)
   phonetic_translit  ~5,000 queries (yaxshi, ucun, cunki, ...)
   russian_calque     ~2,000 queries (Azerbaijani via Russian)
   multi_noise        ~3,000 queries (e.g. diacritic_strip + typo combo)

Run as a background job. Output JSONL is resumable.
"""

import argparse
import asyncio
import json
import logging
import os
import platform
import random
import sys
import time
from collections import Counter
from pathlib import Path

from openai import AsyncOpenAI, OpenAIError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("kimi_strat")

PROJECT_ROOT = Path(__file__).resolve().parent
NAIC_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(NAIC_ROOT))

BASE_URL = "https://integrate.api.nvidia.com/v1"


# ---------------------------------------------------------------------------
# Noise-type-specific schemas and prompts
# ---------------------------------------------------------------------------

STRATIFIED_SCHEMA = {
    "type": "object",
    "properties": {
        "variants": {
            "type": "array",
            "items": {
                "type": "string",
                "description": "One realistic noisy variant of the input."
            },
            "minItems": 1
        }
    },
    "required": ["variants"],
    "additionalProperties": False
}


NOISE_INSTRUCTIONS = {
    "slang": """Generate {n} realistic Azerbaijani SLANG / SMS-style spellings of the query.
This is how real users type in chat or quick searches:
- "necə" -> "neca", "necca", "necesen"
- "üçün" -> "ucun", "uchun"
- "yaxşı" -> "yaxsi", "yaxshi"
- "bilmirəm" -> "bilmirem", "bilmrm"
- "ki" -> "kı", "k"
- doubling vowels for emphasis ("salaaam"), or running words together ("nevar")
Keep meaning identical. Vary noise across variants.""",

    "phonetic_translit": """Generate {n} variants where the query is typed PHONETICALLY by someone without an Azerbaijani keyboard:
- ş -> "sh" or "s" ("şəhər" -> "sheher", "seher")
- ç -> "ch" or "c" ("çay" -> "chay", "cay")
- ğ -> "gh" or "g" ("ağ" -> "agh", "ag")
- x -> "kh" or "h" ("xala" -> "khala")
- ə -> "e" ("evvəl" -> "evvel")
- ü/ö -> "u/o", ı -> "i"
This is the most common form of real-world AZ search noise.""",

    "russian_calque": """Generate {n} variants where the user is typing Azerbaijani words using
Russian-influenced spelling habits (a Russian-speaking Azerbaijani user, no Cyrillic):
- "süd" -> "sut" (Russian-style devoicing)
- "düz" -> "dus", "duz"
- "şəhər" -> "geher", "sheger"
- Keep Latin letters only. No Cyrillic chars.
- Stay in Azerbaijani vocabulary, just with Russian phonetic habits.""",

    "multi_noise": """Generate {n} variants that combine TWO different noise types in each variant
(e.g. diacritic_strip + dropped_letter, or phonetic + missing_space):
- "süd məhsulları" -> "sud mehsullar" (diacritic + dropped letter)
- "yaxşı qiymət" -> "yaxsiqiymet" (phonetic + missing space + diacritic)
- "İlk dəfə" -> "ilkdefe" (capital_i + missing_space + diacritic)
Be realistic — these should look like real users making 2 errors at once.""",
}


def build_system_prompt(noise_type: str, n: int) -> str:
    return f"""You are an Azerbaijani-language data-augmentation assistant.
You generate REALISTIC noisy variants of clean Azerbaijani product-search queries.

Hard rules:
- Preserve meaning exactly.
- Only Azerbaijani Latin letters (ə ş ç ğ x ı i İ ö ü), basic English letters, digits, common punctuation, spaces.
- No Cyrillic except where explicitly stated.
- Each variant must DIFFER from the original.
- Output strict JSON: {{"variants": ["...", "..."]}} — no prose, no markdown.

Today's task:
{NOISE_INSTRUCTIONS[noise_type].format(n=n)}
"""


USER_PROMPT_TEMPLATE = """Clean query: {query}

Return JSON: {{"variants": ["variant1", "variant2", "variant3"]}} with exactly {n} variants."""


# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------

def load_api_keys() -> list[str]:
    keys_path = NAIC_ROOT / "nvidiageneration" / "api_keys.json"
    data = json.load(open(keys_path, "r", encoding="utf-8"))
    keys = []
    for v in data.get("api_keys", {}).values():
        if isinstance(v, list):
            keys.extend(v)
        elif isinstance(v, str):
            keys.append(v)
    return keys


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

async def generate_for(client_picker, bad_keys, model_id, query, noise_type, n_variants,
                       max_retries=6, initial_backoff=2.0):
    sys_prompt = build_system_prompt(noise_type, n_variants)
    user_prompt = USER_PROMPT_TEMPLATE.format(query=query, n=n_variants)
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]
    backoff = initial_backoff
    last_err = None
    for attempt in range(max_retries):
        key_idx, client = client_picker(bad_keys)
        if client is None:
            return None
        try:
            resp = await client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=0.9,
                max_tokens=2000,
                extra_body={"nvext": {"guided_json": STRATIFIED_SCHEMA}},
            )
            content = resp.choices[0].message.content if resp.choices else None
            if not content:
                last_err = "empty"
            else:
                parsed = json.loads(content)
                variants = parsed.get("variants", [])
                cleaned = []
                for t in variants:
                    if isinstance(t, str) and t.strip() and t.strip() != query:
                        cleaned.append(t.strip())
                if cleaned:
                    return cleaned
                last_err = "no valid variants"
        except OpenAIError as e:
            last_err = f"OpenAIError: {e}"
            msg = str(e).lower()
            if "403" in msg or "authorization failed" in msg or "401" in msg:
                bad_keys.add(key_idx)
            elif "429" in msg or "too many requests" in msg or "rate limit" in msg:
                await asyncio.sleep(min(backoff * 4, 30.0))
        except json.JSONDecodeError as e:
            last_err = f"JSONDecode: {e}"
        except Exception as e:
            last_err = f"Other: {e}"
        if attempt < max_retries - 1:
            await asyncio.sleep(min(backoff, 6.0))
            backoff = min(backoff * 1.5, 12.0)
    log.warning(f"FAILED [{noise_type}] {query!r}: {last_err}")
    return None


async def worker(worker_id, queue, client_picker, bad_keys, model_id,
                 out_path, write_lock, counters):
    while True:
        item = await queue.get()
        try:
            if item is None:
                return
            idx, q, nt, n = item["index"], item["query"], item["noise_type"], item["n_variants"]
            variants = await generate_for(client_picker, bad_keys, model_id, q, nt, n)
            success = variants is not None
            counters["done"] += 1
            counters["ok" if success else "fail"] += 1
            counters[nt] = counters.get(nt, 0) + (len(variants) if variants else 0)
            record = {"index": idx, "clean": q, "noise_type": nt,
                      "variants": variants or [], "success": success}
            async with write_lock:
                with open(out_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            log.exception(f"worker {worker_id}: {e}")
        finally:
            queue.task_done()


def already_done(out_path: Path):
    seen = set()
    if not out_path.exists():
        return seen
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                if d.get("success") and d.get("index") is not None:
                    seen.add(d["index"])
            except Exception:
                continue
    return seen


def build_plan(clean_queries, plan: dict[str, int]):
    """plan: {noise_type: num_queries}.
    Returns list of (index, query, noise_type, n_variants) items.
    """
    items = []
    idx = 0
    # filter queries suitable for each type (e.g., russian_calque needs longer words)
    pool = list(clean_queries)
    random.shuffle(pool)
    pool_per_type = {nt: list(pool) for nt in plan}
    for nt, n_queries in plan.items():
        random.shuffle(pool_per_type[nt])
        for q in pool_per_type[nt][:n_queries]:
            items.append({"index": idx, "query": q, "noise_type": nt, "n_variants": 2})
            idx += 1
    random.shuffle(items)
    return items


async def amain(args):
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    keys = load_api_keys()
    log.info(f"Loaded {len(keys)} API keys")
    clients = [AsyncOpenAI(api_key=k, base_url=BASE_URL) for k in keys]
    bad_keys: set = set()
    rr_pos = [0]

    def client_picker(bad):
        if len(bad) >= len(clients):
            return (None, None)
        for _ in range(len(clients)):
            i = rr_pos[0] % len(clients)
            rr_pos[0] += 1
            if i not in bad:
                return (i, clients[i])
        return (None, None)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Load clean queries
    inp = Path(args.input)
    clean = []
    with open(inp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                q = d.get("query")
                if isinstance(q, str) and q.strip():
                    clean.append(q.strip())
            except Exception:
                continue
    log.info(f"Loaded {len(clean):,} clean queries")

    # Build plan
    plan = {
        "phonetic_translit": args.phonetic,
        "slang":             args.slang,
        "russian_calque":    args.russian,
        "multi_noise":       args.multi,
    }
    plan = {k: v for k, v in plan.items() if v > 0}
    items = build_plan(clean, plan)
    log.info(f"Plan: {plan}, total work items: {len(items):,}")

    processed = already_done(out)
    log.info(f"Already done: {len(processed):,}")

    # Async queue
    queue: asyncio.Queue = asyncio.Queue(maxsize=args.workers * 4)
    write_lock = asyncio.Lock()
    counters = {"done": 0, "ok": 0, "fail": 0}

    workers = [
        asyncio.create_task(worker(i, queue, client_picker, bad_keys, args.model,
                                   out, write_lock, counters))
        for i in range(args.workers)
    ]

    enqueued = 0
    start = time.time()
    for item in items:
        if item["index"] in processed:
            continue
        await queue.put(item)
        enqueued += 1
        if enqueued % 500 == 0:
            elapsed = time.time() - start
            rate = counters["done"] / max(elapsed, 1e-6)
            type_breakdown = {nt: counters.get(nt, 0) for nt in plan.keys()}
            log.info(f"enqueued={enqueued}  done={counters['done']}  ok={counters['ok']}  "
                     f"fail={counters['fail']}  rate={rate:.2f}/s  variants={type_breakdown}")

    log.info(f"All enqueued ({enqueued}). Waiting...")
    await queue.join()
    for _ in workers:
        await queue.put(None)
    await asyncio.gather(*workers, return_exceptions=True)

    elapsed = time.time() - start
    log.info(f"DONE in {elapsed:.1f}s. ok={counters['ok']} fail={counters['fail']}")
    for c in clients:
        try:
            await c.close()
        except Exception:
            pass


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/llm_input/clean_queries.jsonl")
    p.add_argument("--output", default="data/llm_output/stratified_pairs.jsonl")
    p.add_argument("--model", default=os.environ.get("NVIDIA_MODEL_ID", "moonshotai/kimi-k2-instruct"))
    p.add_argument("--workers", type=int, default=15)
    p.add_argument("--phonetic", type=int, default=5000)
    p.add_argument("--slang", type=int, default=5000)
    p.add_argument("--russian", type=int, default=2000)
    p.add_argument("--multi", type=int, default=3000)
    return p.parse_args()


def main():
    asyncio.run(amain(parse_args()))


if __name__ == "__main__":
    main()
