"""
Generate training pairs using YOUR smart_noiser with multiple per-category
configurations + the outside-the-box techniques discussed:

  1. Per-category probability profiles    — different noise distributions
  2. Subtle (distance-1..3) variants      — overrides default 4..9 filter
  3. Chained noising (noiser ∘ noiser)    — compound real-world errors
  4. Identity pairs                       — clean == noisy (model learns "do nothing")
  5. Round-trip distillation              — Kimi-corrected real-noise → re-noise with smart_noiser

Source clean queries: data/llm_input/clean_queries.jsonl (76,795 in-domain)
Output:               data/pair/smart_pairs/<config>.json
                      data/pair/smart_pairs/all.json
                      data/pair/smart_pairs/_summary.json
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.smart_noiser import AzerbaijaniTextNoiser

CLEAN_PATH = PROJECT_ROOT / "data" / "llm_input" / "clean_queries.jsonl"
OUT_DIR = PROJECT_ROOT / "data" / "pair" / "smart_pairs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
VOCAB = json.load(open(PROJECT_ROOT / "data" / "pair" / "vocab.json"))
VOCAB_CHARS = set(VOCAB.keys())


# -------------------------------------------------------------------------
# Subclass to control the edit-distance recursion filter per config
# -------------------------------------------------------------------------

class FlexibleNoiser(AzerbaijaniTextNoiser):
    """Same noise pipeline, but configurable edit-distance window.

    The base class re-rolls if final distance is 0, <4, or >9. That is
    great for medium noise but kills all subtle 1-2 char typos, which are
    what real users actually produce most of the time. Here we make the
    window settable so we can build a "subtle" config (1..3) alongside
    medium (4..9) and heavy (5..15) configs.
    """

    def __init__(self, *, min_dist=4, max_dist=9, max_recursion=15, **kwargs):
        super().__init__(**kwargs)
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.max_recursion = max_recursion

    def __call__(self, text):
        original = text
        last_noisy, last_stats = text, None
        for _ in range(self.max_recursion):
            t = original
            if random.random() <= self.eng_prob:
                # eng transformation is "all-or-nothing" so we route here as a short-circuit
                t = self._apply_eng_transformation(t)
                last_stats = self._edit_distance_stats(original, t)
                t = self._apply_space_removal_transformation(t)
                if self.min_dist <= last_stats["sentence_lev_distance"] <= self.max_dist:
                    return t, last_stats
                continue

            if random.random() <= self.rule_based_prob:
                t = self._apply_rule_based_transformation(t)

            if random.random() <= self.weighted_prob:
                t = self._apply_weighted_transformation(t)
            else:
                t = self._apply_adjacent_transformation(t)

            t = self._apply_swap_transformation(t)
            t = self._apply_case_transformation(t)
            t = self._apply_space_removal_transformation(t)
            # Distance computed on the FINAL output (after space removal).
            # Base class computed it before; that hid space-only noise.
            last_stats = self._edit_distance_stats(original, t)
            last_noisy = t
            d = last_stats["sentence_lev_distance"]
            if self.min_dist <= d <= self.max_dist:
                return t, last_stats

        # ran out of tries: return whatever we have if it's at least distinct
        if last_stats and last_stats["sentence_lev_distance"] > 0:
            return last_noisy, last_stats
        return None, last_stats


# -------------------------------------------------------------------------
# Per-category noise configurations
# -------------------------------------------------------------------------

CONFIGS = {
    # Real users without AZ keyboard layout — heavy English/diacritic strip
    "latin_typer_heavy": dict(
        eng_prob=0.85,
        weighted_prob=0.0, rule_based_prob=0.0, adjacent_prob=0.0,
        swap_prob=0.0, case_prob=0.0, remove_space=0.0,
        min_dist=2, max_dist=20,
    ),
    # Mobile thumb typing — adjacent key slips + occasional swaps
    "mobile_typing": dict(
        eng_prob=0.0, weighted_prob=0.0, rule_based_prob=0.0,
        adjacent_prob=0.5, swap_prob=0.06, case_prob=0.005,
        remove_space=0.0,
        min_dist=1, max_dist=5,
    ),
    # Subtle 1-2 char typo — what we missed in previous gen
    "subtle_typo": dict(
        eng_prob=0.0, weighted_prob=0.0, rule_based_prob=0.0,
        adjacent_prob=0.20, swap_prob=0.03, case_prob=0.01,
        remove_space=0.0,
        min_dist=1, max_dist=3,
    ),
    # Phonetic / weighted character substitution (calibrated ə→e/a, ş→sh/w, etc.)
    "phonetic_weighted": dict(
        eng_prob=0.0, weighted_prob=0.85,
        rule_based_prob=0.10, adjacent_prob=0.0, swap_prob=0.0,
        case_prob=0.0, remove_space=0.0,
        min_dist=1, max_dist=10,
    ),
    # Heavy chaos — multiple things wrong
    "heavy_chaotic": dict(
        eng_prob=0.30, weighted_prob=0.60, rule_based_prob=0.40,
        adjacent_prob=0.30, swap_prob=0.02, case_prob=0.01,
        remove_space=0.02,
        min_dist=3, max_dist=15,
    ),
    # Space mistakes — joined or split. Bumped remove_space because
    # queries are short (2-4 words) so 0.05 per-space hardly fires.
    "space_mistakes": dict(
        eng_prob=0.0, weighted_prob=0.0, rule_based_prob=0.0,
        adjacent_prob=0.0, swap_prob=0.0, case_prob=0.0,
        remove_space=0.50,
        min_dist=1, max_dist=4,
    ),
    # Case mistakes — capitalization off
    "case_only": dict(
        eng_prob=0.0, weighted_prob=0.0, rule_based_prob=0.0,
        adjacent_prob=0.0, swap_prob=0.0, case_prob=0.10,
        remove_space=0.0,
        min_dist=1, max_dist=8,
    ),
}

# Per-config target sample counts
TARGETS = {
    "latin_typer_heavy":  20000,
    "mobile_typing":      15000,
    "subtle_typo":        15000,
    "phonetic_weighted":  18000,
    "heavy_chaotic":      10000,
    "space_mistakes":      6000,
    "case_only":           4000,
    "chained":             8000,   # special: noiser ∘ noiser
    "identity":           15000,   # clean == noisy
}


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def valid_pair(clean, noisy):
    if not isinstance(clean, str) or not isinstance(noisy, str):
        return False
    if not clean.strip() or not noisy.strip():
        return False
    if len(noisy) > max(2 * len(clean), 60):
        return False
    if any(c not in VOCAB_CHARS for c in clean):
        return False
    if any(c not in VOCAB_CHARS for c in noisy):
        return False
    return True


def load_queries():
    qs = []
    with open(CLEAN_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                q = d.get("query")
                if isinstance(q, str) and q.strip():
                    qs.append(q.strip())
            except Exception:
                continue
    return qs


def generate_for_config(config_name, config_kwargs, queries, target):
    noiser = FlexibleNoiser(**config_kwargs)
    out = []
    seen = set()
    attempts = 0
    max_attempts = target * 6
    while len(out) < target and attempts < max_attempts:
        attempts += 1
        q = random.choice(queries)
        try:
            noisy, _stats = noiser(q)
        except Exception:
            continue
        if noisy is None:
            continue
        if noisy == q:
            continue
        if not valid_pair(q, noisy):
            continue
        key = (q, noisy)
        if key in seen:
            continue
        seen.add(key)
        out.append({"clean": q, "noisy": noisy})
    return out, attempts


def generate_chained(queries, target):
    """Apply two noisers in sequence to produce compound real-world errors."""
    n1 = FlexibleNoiser(**{**CONFIGS["mobile_typing"], "min_dist": 1, "max_dist": 6})
    n2 = FlexibleNoiser(**{**CONFIGS["phonetic_weighted"], "min_dist": 1, "max_dist": 6})
    out, seen, attempts = [], set(), 0
    max_attempts = target * 6
    while len(out) < target and attempts < max_attempts:
        attempts += 1
        q = random.choice(queries)
        try:
            mid, _ = n1(q)
            if mid is None or mid == q:
                continue
            final, _ = n2(mid)
            if final is None or final == mid:
                # Apply second noiser to original instead if chaining failed
                final, _ = n2(q)
                if final is None or final == q:
                    continue
        except Exception:
            continue
        if not valid_pair(q, final):
            continue
        key = (q, final)
        if key in seen:
            continue
        seen.add(key)
        out.append({"clean": q, "noisy": final})
    return out, attempts


def generate_identity(queries, target):
    pool = random.sample(queries, min(len(queries), target))
    return [{"clean": q, "noisy": q} for q in pool if valid_pair(q, q)]


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    print("Loading clean queries...")
    queries = load_queries()
    print(f"  {len(queries):,} queries available\n")

    all_pairs = []
    summary = {}

    for name in TARGETS:
        target = TARGETS[name]
        if name == "chained":
            pairs, attempts = generate_chained(queries, target)
        elif name == "identity":
            pairs = generate_identity(queries, target)
            attempts = len(pairs)
        else:
            pairs, attempts = generate_for_config(name, CONFIGS[name], queries, target)
        path = OUT_DIR / f"{name}.json"
        json.dump(pairs, open(path, "w", encoding="utf-8"), ensure_ascii=False)
        all_pairs.extend(pairs)
        summary[name] = {"target": target, "produced": len(pairs), "attempts": attempts}
        print(f"  {name:<22} target={target:>6,}  produced={len(pairs):>6,}  attempts={attempts:>6,}")

    random.shuffle(all_pairs)
    json.dump(all_pairs, open(OUT_DIR / "all.json", "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(summary, open(OUT_DIR / "_summary.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print(f"\nTotal pairs: {len(all_pairs):,}  ->  {OUT_DIR/'all.json'}")
    print("\nSample per config:")
    for name in TARGETS:
        data = json.load(open(OUT_DIR / f"{name}.json"))
        if data:
            ex = random.choice(data)
            c = ex["clean"][:50]
            n = ex["noisy"][:50]
            print(f"  {name:<22}  clean: {c!r:<55}  noisy: {n!r}")


if __name__ == "__main__":
    main()
