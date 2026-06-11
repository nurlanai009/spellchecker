"""
Local rule-based + identity pair generators — no API calls, fast.

Applies to the 76,795 clean queries we extracted from in-domain sources.
Produces per-noise-type JSON files plus a combined file. Quantities are
sampled to MATCH the target distribution defined below.

Target distribution (calibrated for real-world Azerbaijani search-box typing):
  identity              12%
  diacritic_strip       25%
  typo_keyboard         15%
  missing_space          8%
  extra_space            4%
  capital_i_confusion    8%
  dropped_letter         3%
  doubled_letter         3%
  mixed_case             3%

Other categories (phonetic_translit, slang, russian_calque, digit_letter,
multi_noise) are produced by the LLM. The LLM-free portion = ~80% of pairs.

Output: data/pair/local_pairs/<type>.json and data/pair/local_pairs/all.json
"""

import json
import random
import re
from collections import Counter
from pathlib import Path

random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parent
CLEAN_PATH = PROJECT_ROOT / "data" / "llm_input" / "clean_queries.jsonl"
OUT_DIR = PROJECT_ROOT / "data" / "pair" / "local_pairs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
VOCAB_PATH = PROJECT_ROOT / "data" / "pair" / "vocab.json"

# Target counts per type (calibrated against the 80% non-LLM share)
TARGETS = {
    "identity":              15000,
    "diacritic_strip":       30000,
    "typo_keyboard":         18000,
    "missing_space":         10000,
    "extra_space":            5000,
    "capital_i_confusion":   10000,
    "dropped_letter":         4000,
    "doubled_letter":         4000,
    "mixed_case":             4000,
}

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

DIACRITIC_MAP = {
    "ə": "e", "Ə": "E",
    "ş": "s", "Ş": "S",
    "ç": "c", "Ç": "C",
    "ğ": "g", "Ğ": "G",
    "ı": "i", "İ": "I",
    "ö": "o", "Ö": "O",
    "ü": "u", "Ü": "U",
}
DIACRITIC_SH_CH_GH = {
    "ş": "sh", "Ş": "SH",
    "ç": "ch", "Ç": "CH",
    "ğ": "gh", "Ğ": "GH",
}

# Simplified QWERTY adjacency
ADJ = {
    "q": "üwa", "w": "qse", "e": "wsd", "r": "etf", "t": "ryg", "y": "tuh",
    "u": "yij", "i": "uok", "o": "ipl", "p": "oöü",
    "a": "qsz", "s": "adwxz", "d": "sfxce", "f": "dgcvr", "g": "fhbvt",
    "h": "gjnby", "j": "hkmnu", "k": "jlmi", "l": "kö",
    "z": "xs", "x": "zsdc", "c": "xdvf", "v": "cbgf", "b": "vgnh", "n": "bjmh", "m": "njk",
    "ə": "wae", "ş": "sxc", "ç": "vbn", "ğ": "qw", "ö": "öpl", "ü": "üpa", "ı": "uo",
}


def strip_diacritics(text: str, prob: float = 0.85) -> str:
    """Replace AZ diacritics with ASCII. Sometimes use sh/ch/gh forms."""
    out = []
    for ch in text:
        if ch in DIACRITIC_MAP and random.random() < prob:
            # 25% chance to use sh/ch/gh form for the relevant letters
            if ch in DIACRITIC_SH_CH_GH and random.random() < 0.25:
                out.append(DIACRITIC_SH_CH_GH[ch])
            else:
                out.append(DIACRITIC_MAP[ch])
        else:
            out.append(ch)
    return "".join(out)


def keyboard_typo(text: str, n_changes: int = None) -> str:
    if not text:
        return text
    if n_changes is None:
        n_changes = 1 if len(text) <= 6 else random.choice([1, 1, 2])
    chars = list(text)
    indices = list(range(len(chars)))
    random.shuffle(indices)
    changed = 0
    for i in indices:
        ch = chars[i]
        lo = ch.lower()
        if lo in ADJ:
            repl = random.choice(ADJ[lo])
            chars[i] = repl.upper() if ch.isupper() else repl
            changed += 1
            if changed >= n_changes:
                break
    return "".join(chars)


def missing_space(text: str) -> str:
    toks = text.split()
    if len(toks) < 2:
        return text
    i = random.randrange(len(toks) - 1)
    joined = toks[i] + toks[i + 1]
    return " ".join(toks[:i] + [joined] + toks[i + 2:])


def extra_space(text: str) -> str:
    toks = text.split()
    out = []
    for t in toks:
        if len(t) >= 6 and random.random() < 0.4:
            i = random.randrange(2, len(t) - 2)
            out.append(t[:i] + " " + t[i:])
        else:
            out.append(t)
    if " ".join(out) == text and toks:
        # force at least one split on a longish word
        t = max(toks, key=len)
        if len(t) >= 4:
            i = random.randrange(2, len(t) - 1)
            text2 = text.replace(t, t[:i] + " " + t[i:], 1)
            return text2
    return " ".join(out)


def capital_i_swap(text: str) -> str:
    """Replace İ at word starts with Latin I; sometimes ı↔i inside words."""
    new = re.sub(r"\bİ", "I", text)
    if new != text:
        return new
    # fall back: lowercase first letter of capitalized first word
    if text and text[0].isupper():
        return text[0].lower() + text[1:]
    return text


def dropped_letter(text: str) -> str:
    if len(text) < 4:
        return text
    # pick a non-edge position, not on whitespace
    indices = [i for i, c in enumerate(text) if i not in (0, len(text) - 1) and not c.isspace()]
    if not indices:
        return text
    i = random.choice(indices)
    return text[:i] + text[i + 1:]


def doubled_letter(text: str) -> str:
    if len(text) < 2:
        return text
    indices = [i for i, c in enumerate(text) if c.isalpha()]
    if not indices:
        return text
    i = random.choice(indices)
    return text[:i + 1] + text[i] + text[i + 1:]


def mixed_case(text: str) -> str:
    # toggle 1-3 character cases at random
    chars = list(text)
    alpha_idx = [i for i, c in enumerate(chars) if c.isalpha()]
    if not alpha_idx:
        return text
    k = min(random.randint(1, 3), len(alpha_idx))
    for i in random.sample(alpha_idx, k):
        chars[i] = chars[i].swapcase()
    return "".join(chars)


GENERATORS = {
    "identity":             lambda q: q,
    "diacritic_strip":      strip_diacritics,
    "typo_keyboard":        keyboard_typo,
    "missing_space":        missing_space,
    "extra_space":          extra_space,
    "capital_i_confusion":  capital_i_swap,
    "dropped_letter":       dropped_letter,
    "doubled_letter":       doubled_letter,
    "mixed_case":           mixed_case,
}


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def load_clean_queries():
    out = []
    with open(CLEAN_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                q = d.get("query")
                if isinstance(q, str) and q.strip():
                    out.append(q.strip())
            except Exception:
                continue
    return out


def main():
    print("Loading clean queries...")
    queries = load_clean_queries()
    print(f"  {len(queries):,} queries available")

    vocab = json.load(open(VOCAB_PATH, "r", encoding="utf-8"))
    vocab_chars = set(vocab.keys())

    all_pairs = []
    summary = {}

    for noise_type, target in TARGETS.items():
        gen_fn = GENERATORS[noise_type]
        produced = []
        attempts = 0
        max_attempts = target * 8
        # For capital_i: only queries that contain İ are eligible
        if noise_type == "capital_i_confusion":
            eligible = [q for q in queries if "İ" in q or any(w[:1].isupper() for w in q.split())]
        elif noise_type in ("missing_space", "extra_space"):
            eligible = [q for q in queries if " " in q]
        else:
            eligible = queries

        if not eligible:
            print(f"  {noise_type}: no eligible queries, skipped")
            summary[noise_type] = 0
            continue

        seen = set()
        while len(produced) < target and attempts < max_attempts:
            attempts += 1
            q = random.choice(eligible)
            try:
                noisy = gen_fn(q)
            except Exception:
                continue
            if noise_type != "identity" and noisy == q:
                continue
            if not noisy:
                continue
            # vocab filter
            if any(c not in vocab_chars for c in noisy):
                continue
            if any(c not in vocab_chars for c in q):
                continue
            key = (q, noisy)
            if key in seen:
                continue
            seen.add(key)
            produced.append({"clean": q, "noisy": noisy})

        out_path = OUT_DIR / f"{noise_type}.json"
        json.dump(produced, open(out_path, "w", encoding="utf-8"), ensure_ascii=False)
        all_pairs.extend(produced)
        summary[noise_type] = len(produced)
        print(f"  {noise_type:<25} {len(produced):>6,} pairs (attempts: {attempts:,})")

    random.shuffle(all_pairs)
    combined = OUT_DIR / "all.json"
    json.dump(all_pairs, open(combined, "w", encoding="utf-8"), ensure_ascii=False)

    summary_path = OUT_DIR / "_summary.json"
    json.dump(
        {"target": TARGETS, "produced": summary, "total": len(all_pairs)},
        open(summary_path, "w", encoding="utf-8"),
        indent=2,
        ensure_ascii=False,
    )

    print(f"\nTotal local pairs: {len(all_pairs):,} -> {combined}")
    print(f"Summary: {summary_path}")

    # show a sample per type
    print("\nSamples:")
    for nt in TARGETS:
        path = OUT_DIR / f"{nt}.json"
        if not path.exists():
            continue
        data = json.load(open(path))
        if data:
            ex = random.choice(data)
            print(f"  {nt:<25} clean: {ex['clean'][:50]!r:<50}  noisy: {ex['noisy'][:50]!r}")


if __name__ == "__main__":
    main()
