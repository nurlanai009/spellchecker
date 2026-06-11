"""
Mix all data sources to the target distribution for v3 training.

Inputs:
  - data/pair/local_pairs/*.json                  (100k local rule-based + identity)
  - data/llm_output/noisy_pairs_full.jsonl        (free-pick Kimi gen → 13k)
  - data/llm_output/stratified_pairs.jsonl        (stratified Kimi → 22k)
  - data/pair/use_cases/all_use_cases.json        (33k balanced edge cases)
  - data/pair/train/200k_words_train_noised.json  (163k word-level)
  - data/pair/train/train_sentences_azeri.json    (109k sentences)
  - data/pair/train/targeted_pairs.json           (16k capital-İ targeted)

Target final mix size: ~180k pairs.
Weights chosen to match real-world Azerbaijani search-box typing distribution.

Outputs:
  data/pair/train_v3/all.json             (the mixed training set, ready for training)
  data/pair/train_v3/_mix_summary.json    (counts per source)
"""

import json
import random
from collections import Counter
from pathlib import Path

random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parent
VOCAB = json.load(open(PROJECT_ROOT / "data" / "pair" / "vocab.json"))
VOCAB_CHARS = set(VOCAB.keys())

OUT_DIR = PROJECT_ROOT / "data" / "pair" / "train_v4"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_for_compare(s: str) -> str:
    return " ".join(s.strip().lower().split())


def valid_pair(p, max_len=120):
    if not isinstance(p, dict):
        return False
    c = p.get("clean")
    n = p.get("noisy")
    if not isinstance(c, str) or not isinstance(n, str):
        return False
    c, n = c.strip(), n.strip()
    if not c or not n:
        return False
    if len(c) > max_len or len(n) > max_len:
        return False
    if any(ch not in VOCAB_CHARS for ch in c):
        return False
    if any(ch not in VOCAB_CHARS for ch in n):
        return False
    return True


def load_json_array(path):
    if not path.exists():
        return []
    try:
        data = json.load(open(path, "r", encoding="utf-8"))
        return [p for p in data if valid_pair(p)]
    except Exception as e:
        print(f"  ERROR {path}: {e}")
        return []


def load_freepick_jsonl(path):
    """Schema: {clean, variants:[{text, noise_type}], success}"""
    out = []
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if not rec.get("success"):
                continue
            clean = rec.get("clean", "").strip()
            for v in rec.get("variants", []) or []:
                if isinstance(v, dict):
                    t = (v.get("text") or "").strip()
                    if t:
                        pair = {"clean": clean, "noisy": t}
                        if valid_pair(pair) and normalize_for_compare(clean) != normalize_for_compare(t):
                            out.append(pair)
    return out


def load_stratified_jsonl(path):
    """Schema: {clean, noise_type, variants:[str], success}"""
    out = []
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if not rec.get("success"):
                continue
            clean = rec.get("clean", "").strip()
            for t in rec.get("variants", []) or []:
                if isinstance(t, str):
                    t = t.strip()
                    if t:
                        pair = {"clean": clean, "noisy": t}
                        if valid_pair(pair) and normalize_for_compare(clean) != normalize_for_compare(t):
                            out.append(pair)
    return out


# ---------------------------------------------------------------------------
# Sources with sampling caps
# ---------------------------------------------------------------------------

SOURCES = [
    # name, loader, path, target sample size
    # ----- smart_noiser pairs (calibrated to AZ patterns, replaces old local_pairs) -----
    ("smart_identity",        "local",    PROJECT_ROOT / "data/pair/smart_pairs/identity.json",            15000),
    ("smart_latin_typer",     "local",    PROJECT_ROOT / "data/pair/smart_pairs/latin_typer_heavy.json",   20000),
    ("smart_mobile_typing",   "local",    PROJECT_ROOT / "data/pair/smart_pairs/mobile_typing.json",       12000),
    ("smart_subtle_typo",     "local",    PROJECT_ROOT / "data/pair/smart_pairs/subtle_typo.json",         12000),
    ("smart_phonetic",        "local",    PROJECT_ROOT / "data/pair/smart_pairs/phonetic_weighted.json",   15000),
    ("smart_heavy_chaotic",   "local",    PROJECT_ROOT / "data/pair/smart_pairs/heavy_chaotic.json",        8000),
    ("smart_space_mistakes",  "local",    PROJECT_ROOT / "data/pair/smart_pairs/space_mistakes.json",       5000),
    ("smart_case_only",       "local",    PROJECT_ROOT / "data/pair/smart_pairs/case_only.json",            3000),
    ("smart_chained",         "local",    PROJECT_ROOT / "data/pair/smart_pairs/chained.json",              7000),
    # ----- LLM pairs (Kimi K2) -----
    ("kimi_freepick",         "freepick", PROJECT_ROOT / "data/llm_output/noisy_pairs_full.jsonl",          8000),
    ("kimi_stratified",       "stratified", PROJECT_ROOT / "data/llm_output/stratified_pairs.jsonl",       20000),
    # ----- existing curated sources -----
    ("use_cases",             "local",    PROJECT_ROOT / "data/pair/use_cases/all_use_cases.json",         12000),
    ("words_noised_200k",     "local",    PROJECT_ROOT / "data/pair/train/200k_words_train_noised.json",   20000),
    ("sentences_azeri",       "local",    PROJECT_ROOT / "data/pair/train/train_sentences_azeri.json",     15000),
    ("targeted_capital_i",    "local",    PROJECT_ROOT / "data/pair/train/targeted_pairs.json",             5000),
]


def main():
    print("Loading and mixing sources...\n")

    all_pairs = []
    summary = []
    pair_seen = set()

    for name, kind, path, target in SOURCES:
        if kind == "local":
            pool = load_json_array(path)
        elif kind == "freepick":
            pool = load_freepick_jsonl(path)
        elif kind == "stratified":
            pool = load_stratified_jsonl(path)
        else:
            continue

        # de-dup within pool
        seen_local = set()
        deduped = []
        for p in pool:
            key = (p["clean"], p["noisy"])
            if key in seen_local:
                continue
            seen_local.add(key)
            deduped.append(p)

        random.shuffle(deduped)
        sampled = deduped[:target]

        # de-dup against global
        added = 0
        for p in sampled:
            key = (p["clean"], p["noisy"])
            if key in pair_seen:
                continue
            pair_seen.add(key)
            all_pairs.append(p)
            added += 1

        summary.append({"name": name, "available": len(deduped), "target": target, "added": added})
        print(f"  {name:<28} avail={len(deduped):>7,}  target={target:>6,}  added={added:>6,}")

    random.shuffle(all_pairs)

    # global stats
    lens_clean = [len(p["clean"]) for p in all_pairs]
    lens_noisy = [len(p["noisy"]) for p in all_pairs]
    identity = sum(1 for p in all_pairs if p["clean"] == p["noisy"])

    out_path = OUT_DIR / "all.json"
    json.dump(all_pairs, open(out_path, "w", encoding="utf-8"), ensure_ascii=False)

    summary_path = OUT_DIR / "_mix_summary.json"
    json.dump({
        "total_pairs": len(all_pairs),
        "identity_pairs": identity,
        "identity_pct": round(100 * identity / max(1, len(all_pairs)), 2),
        "clean_len_min": min(lens_clean) if lens_clean else 0,
        "clean_len_max": max(lens_clean) if lens_clean else 0,
        "clean_len_mean": round(sum(lens_clean) / max(1, len(lens_clean)), 1),
        "sources": summary,
    }, open(summary_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print(f"\n=== FINAL MIXED TRAINING SET ===")
    print(f"  Total pairs:        {len(all_pairs):,}")
    print(f"  Identity (do-nothing): {identity:,} ({100*identity/max(1,len(all_pairs)):.1f}%)")
    print(f"  Clean length: min={min(lens_clean)} max={max(lens_clean)} mean={sum(lens_clean)/max(1,len(lens_clean)):.1f}")
    print(f"\n  Wrote: {out_path}")
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
