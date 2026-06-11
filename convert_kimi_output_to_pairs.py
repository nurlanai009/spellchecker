"""
Convert the JSONL output of generate_noisy_pairs_kimi.py into the
spellchecker training-pair format:

  input:   data/llm_output/noisy_pairs_full.jsonl
           lines like:
             {"index": N, "clean": "...", "variants": [{"text": "...", "noise_type": "..."}, ...], "success": true}

  outputs:
    data/pair/train/kimi_pairs.json           - all (clean, noisy) pairs
    data/llm_output/by_noise_type/<type>.json - per-noise-type breakdown for inspection
    data/llm_output/_kimi_summary.json        - counts and quality stats

Quality filters applied:
  - Drop variants identical to the clean (no actual noise)
  - Drop variants with chars not in vocab.json
  - Drop variants > 2x the clean length or < 1 char
  - Drop variants where clean == noisy after whitespace normalization
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
IN_PATH = PROJECT_ROOT / "data" / "llm_output" / "noisy_pairs_full.jsonl"
OUT_PAIRS = PROJECT_ROOT / "data" / "pair" / "train" / "kimi_pairs.json"
OUT_BY_TYPE_DIR = PROJECT_ROOT / "data" / "llm_output" / "by_noise_type"
OUT_SUMMARY = PROJECT_ROOT / "data" / "llm_output" / "_kimi_summary.json"
VOCAB_PATH = PROJECT_ROOT / "data" / "pair" / "vocab.json"


def normalize_for_compare(s: str) -> str:
    return " ".join(s.strip().lower().split())


def main():
    OUT_BY_TYPE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PAIRS.parent.mkdir(parents=True, exist_ok=True)

    vocab = json.load(open(VOCAB_PATH, "r", encoding="utf-8"))
    vocab_chars = set(vocab.keys())

    by_type: dict[str, list[dict]] = defaultdict(list)
    counters = Counter()
    pair_seen = set()

    if not IN_PATH.exists():
        print(f"ERROR: input not found: {IN_PATH}")
        return

    with open(IN_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                counters["bad_json"] += 1
                continue

            if not rec.get("success"):
                counters["failed_query"] += 1
                continue

            clean = (rec.get("clean") or "").strip()
            if not clean:
                counters["empty_clean"] += 1
                continue

            for v in rec.get("variants", []) or []:
                if not isinstance(v, dict):
                    counters["bad_variant_shape"] += 1
                    continue
                text = (v.get("text") or "").strip()
                nt = (v.get("noise_type") or "unknown").strip()

                if not text:
                    counters["empty_text"] += 1
                    continue
                if normalize_for_compare(text) == normalize_for_compare(clean):
                    counters["identical_to_clean"] += 1
                    continue
                if len(text) < 1 or len(text) > max(2 * len(clean), 60):
                    counters["bad_length"] += 1
                    continue

                # vocab filter
                bad = [c for c in text if c not in vocab_chars]
                if bad:
                    counters["oov_char"] += 1
                    continue
                bad_c = [c for c in clean if c not in vocab_chars]
                if bad_c:
                    counters["oov_char_clean"] += 1
                    continue

                pair_key = (clean, text)
                if pair_key in pair_seen:
                    counters["duplicate_pair"] += 1
                    continue
                pair_seen.add(pair_key)

                pair = {"clean": clean, "noisy": text, "noise_type": nt}
                by_type[nt].append(pair)
                counters["accepted"] += 1

    # write per-type
    type_summary = {}
    for nt, items in by_type.items():
        # strip noise_type before saving as training data (training format is just clean/noisy)
        items_for_training = [{"clean": p["clean"], "noisy": p["noisy"]} for p in items]
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", nt) or "unknown"
        path = OUT_BY_TYPE_DIR / f"{safe_name}.json"
        json.dump(items_for_training, open(path, "w", encoding="utf-8"), ensure_ascii=False)
        type_summary[nt] = len(items_for_training)

    # combined training file: clean/noisy only
    all_pairs = []
    for items in by_type.values():
        for p in items:
            all_pairs.append({"clean": p["clean"], "noisy": p["noisy"]})
    json.dump(all_pairs, open(OUT_PAIRS, "w", encoding="utf-8"), ensure_ascii=False)

    summary = {
        "input": str(IN_PATH),
        "accepted_pairs": counters["accepted"],
        "rejection_counts": dict(counters),
        "pairs_by_noise_type": type_summary,
        "output_combined": str(OUT_PAIRS),
        "output_per_type_dir": str(OUT_BY_TYPE_DIR),
    }
    json.dump(summary, open(OUT_SUMMARY, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print("=" * 60)
    print(f"Accepted pairs: {counters['accepted']:,}")
    print("Rejections:")
    for k, v in counters.most_common():
        if k != "accepted":
            print(f"  {k:<25} {v:,}")
    print("\nBy noise type:")
    for nt, c in sorted(type_summary.items(), key=lambda x: -x[1]):
        print(f"  {nt:<25} {c:,}")
    print(f"\nWrote combined pairs:  {OUT_PAIRS}")
    print(f"Wrote per-type files:  {OUT_BY_TYPE_DIR}")
    print(f"Wrote summary:         {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
