"""
Generate targeted training pairs to address failure modes found in the eval:
  1) Capital "I" / "İ" confusion at word start (model emitted <UNK> for İ).
  2) Sentence-level identity examples (so the model learns "do nothing"
     when input is already clean).
  3) Diacritic-stripping (sh, ch, gh, e for ə) — common typing-without-AZ-keyboard.

Output:
  data/pair/train/targeted_pairs.json  (list of {"clean": ..., "noisy": ...})
"""

import json
import random
import re
from collections import Counter
from pathlib import Path

random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parent

# ---- 1. Harvest İ-initial words from existing clean corpora ----------

SOURCES = [
    PROJECT_ROOT / "data" / "clean" / "200k_words_train.json",
    PROJECT_ROOT / "data" / "pair" / "train" / "train_sentences_azeri.json",
    PROJECT_ROOT / "data" / "pair" / "train" / "train_words_azeri.json",
]


def iter_words(path: Path):
    data = json.load(open(path, "r", encoding="utf-8"))
    for item in data:
        if isinstance(item, str):
            text = item
        elif isinstance(item, dict):
            text = item.get("clean") or item.get("original") or item.get("text") or ""
        else:
            continue
        for w in re.split(r"[\s,.!?;:\"'()\[\]/—–-]+", text):
            if w:
                yield w


def harvest_capital_i_words():
    counter = Counter()
    for src in SOURCES:
        if not src.exists():
            continue
        for w in iter_words(src):
            if w[0:1] == "İ" and len(w) <= 30:
                counter[w] += 1
    return counter


# ---- 2. Capital İ noise patterns ----------

def latinize_first_letter(word: str) -> str:
    """İslam -> Islam (most common human error: typing Latin I instead of İ)."""
    return "I" + word[1:]


def degrade_diacritics(word: str) -> str:
    """Random replacement of AZ diacritics with ASCII equivalents."""
    table = {
        "ə": "e", "Ə": "E",
        "ı": "i", "İ": "I",
        "ö": "o", "Ö": "O",
        "ü": "u", "Ü": "U",
        "ç": "c", "Ç": "C",
        "ş": "s", "Ş": "S",
        "ğ": "g", "Ğ": "G",
    }
    out = []
    for ch in word:
        if ch in table and random.random() < 0.6:
            out.append(table[ch])
        else:
            out.append(ch)
    return "".join(out)


# ---- 3. Build the three datasets ----------

def make_capital_i_pairs(words_counter, total=8000):
    """
    For each frequent İ-initial word, produce several pair variants:
      - "Islam"      <- "İslam"         (Latinized İ only)
      - "islam"      <- "İslam"         (lowercased)
      - "i̇slam"      <- "İslam"         (combining-dot form, sometimes seen)
      - degraded     <- "İslam"
    """
    pairs = []
    # weight by frequency so we don't oversample rare names
    pool = list(words_counter.items())
    pool.sort(key=lambda x: -x[1])
    pool = pool[:5000]  # top 5000 distinct İ-initial words

    if not pool:
        return pairs

    while len(pairs) < total:
        w, _ = random.choice(pool)
        # variant 1: Latin capital I
        pairs.append({"clean": w, "noisy": latinize_first_letter(w)})
        # variant 2: lowercase first letter
        if len(pairs) < total:
            pairs.append({"clean": w, "noisy": w.lower()})
        # variant 3: combining-dot form (̇ = combining dot above)
        if len(pairs) < total and random.random() < 0.3:
            pairs.append({"clean": w, "noisy": "İ" + w[1:]})
        # variant 4: full diacritic strip
        if len(pairs) < total and random.random() < 0.5:
            stripped = degrade_diacritics(w)
            if stripped != w:
                pairs.append({"clean": w, "noisy": stripped})
    return pairs[:total]


def make_identity_sentence_pairs(total=3000):
    """Sentence-level identity examples so the model learns
    to leave correct text alone."""
    src = PROJECT_ROOT / "data" / "pair" / "train" / "train_sentences_azeri.json"
    if not src.exists():
        return []
    sentences = []
    data = json.load(open(src))
    for d in data:
        c = d.get("clean")
        if not c:
            continue
        if len(c) <= 120:
            sentences.append(c)
    random.shuffle(sentences)
    pairs = [{"clean": s, "noisy": s} for s in sentences[:total]]
    return pairs


def make_diacritic_strip_pairs(total=5000):
    """Words/short phrases where the noisy version strips Azerbaijani diacritics
    (very common in real-world typing)."""
    src = PROJECT_ROOT / "data" / "clean" / "200k_words_train.json"
    if not src.exists():
        return []
    words = json.load(open(src))
    words = [w for w in words if isinstance(w, str) and 3 <= len(w) <= 30]

    pairs = []
    random.shuffle(words)
    for w in words:
        if len(pairs) >= total:
            break
        noisy = degrade_diacritics(w)
        if noisy != w:
            pairs.append({"clean": w, "noisy": noisy})
    return pairs


def main():
    print("Harvesting İ-initial words ...")
    words = harvest_capital_i_words()
    print(f"  found {len(words):,} distinct İ-initial words (top 10: {words.most_common(10)})")

    print("Building Capital-İ confusion pairs ...")
    cap_pairs = make_capital_i_pairs(words, total=8000)
    print(f"  generated {len(cap_pairs):,}")

    print("Building identity sentence pairs ...")
    id_pairs = make_identity_sentence_pairs(total=3000)
    print(f"  generated {len(id_pairs):,}")

    print("Building diacritic-strip pairs ...")
    diac_pairs = make_diacritic_strip_pairs(total=5000)
    print(f"  generated {len(diac_pairs):,}")

    all_pairs = cap_pairs + id_pairs + diac_pairs
    random.shuffle(all_pairs)

    out_path = PROJECT_ROOT / "data" / "pair" / "train" / "targeted_pairs.json"
    json.dump(all_pairs, open(out_path, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"\nWrote {len(all_pairs):,} targeted pairs to {out_path}")

    # Show a few samples
    print("\nSamples:")
    for p in random.sample(all_pairs, min(15, len(all_pairs))):
        print(f"  clean: {p['clean']!r:<40}  noisy: {p['noisy']!r}")


if __name__ == "__main__":
    main()
