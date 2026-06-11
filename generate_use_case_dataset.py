"""
Generate a comprehensive Azerbaijani spellchecker dataset covering all the
common use cases the model must handle.

Each category is written to data/pair/use_cases/<n>_<name>.json as a list of
{"clean": str, "noisy": str} pairs. A combined file all_use_cases.json
concatenates everything for direct training.

Categories:
   1 identity                — input is already correct
   2 diacritic_strip         — ə/ş/ç/ğ/ı/ü/ö → ASCII equivalents
   3 capital_i               — İ↔I confusion at word starts
   4 keyboard_adjacent       — QWERTY adjacent-key typos
   5 double_letters          — accidentally repeated character
   6 dropped_letters         — accidentally deleted character
   7 letter_swap             — adjacent character transposition
   8 case_errors             — uppercase/lowercase fixes
   9 missing_space           — "necəyazırsan" → "necə yazırsan"
  10 extra_space             — "qızıl gül" → "qızılgül" (joined compounds)
  11 russian_translit        — Azerbaijani written with Cyrillic letters
  12 turkish_words           — Turkish word forms used instead of AZ
  13 english_borrowings      — English words used instead of AZ equivalents
  14 digit_letter_confusion  — 0↔o, 1↔l/i, 3↔ə, 5↔s, 8↔B (phone keypad style)
  15 smart_quotes            — typographic quotes / dashes normalization
  16 slang_colloquial        — common spoken/SMS-style spellings
  17 repeated_letters        — "salaaam" → "salam"
  18 proper_nouns            — names with İ/Ə/Ş written without diacritics
  19 numbers_units           — "2 .4ghz", "100km/saat" spacing
  20 missing_diacritic_only  — single-diacritic typo cases (ə→e, ş→s, ...)

Run:
    python generate_use_case_dataset.py
"""

import json
import os
import random
import re
import string
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple

random.seed(7)

PROJECT_ROOT = Path(__file__).resolve().parent
OUT_DIR = PROJECT_ROOT / "data" / "pair" / "use_cases"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Source corpora
# ---------------------------------------------------------------------------

WORD_SOURCES = [
    PROJECT_ROOT / "data" / "clean" / "200k_words_train.json",
    PROJECT_ROOT / "data" / "clean" / "200k_words_test.json",
]
SENTENCE_SOURCES = [
    PROJECT_ROOT / "data" / "pair" / "train" / "train_sentences_azeri.json",
    PROJECT_ROOT / "data" / "pair" / "test" / "test_sentences_azeri.json",
]


def load_words() -> List[str]:
    words = []
    for p in WORD_SOURCES:
        if not p.exists():
            continue
        data = json.load(open(p, "r", encoding="utf-8"))
        for w in data:
            if isinstance(w, str) and 2 <= len(w) <= 25 and w.strip():
                words.append(w)
    # dedupe preserving order
    seen = set(); uniq = []
    for w in words:
        if w not in seen:
            uniq.append(w); seen.add(w)
    return uniq


def load_sentences(max_len=110, min_len=8) -> List[str]:
    sents = []
    for p in SENTENCE_SOURCES:
        if not p.exists():
            continue
        data = json.load(open(p, "r", encoding="utf-8"))
        for d in data:
            c = d.get("clean") if isinstance(d, dict) else d
            if isinstance(c, str) and min_len <= len(c) <= max_len:
                sents.append(c.strip())
    seen = set(); uniq = []
    for s in sents:
        if s not in seen:
            uniq.append(s); seen.add(s)
    return uniq


# ---------------------------------------------------------------------------
# 1. Identity (input already correct)
# ---------------------------------------------------------------------------

def gen_identity(words: List[str], sents: List[str], n: int = 5000) -> List[Dict]:
    pairs = []
    word_pool = random.sample(words, min(len(words), n // 2))
    sent_pool = random.sample(sents, min(len(sents), n // 2))
    for w in word_pool:
        pairs.append({"clean": w, "noisy": w})
    for s in sent_pool:
        pairs.append({"clean": s, "noisy": s})
    random.shuffle(pairs)
    return pairs[:n]


# ---------------------------------------------------------------------------
# 2. Diacritic strip (full)
# ---------------------------------------------------------------------------

DIACRITIC_MAP = {
    "ə": "e", "Ə": "E",
    "ş": "s", "Ş": "S",
    "ç": "c", "Ç": "C",
    "ğ": "g", "Ğ": "G",
    "ı": "i", "İ": "I",
    "ö": "o", "Ö": "O",
    "ü": "u", "Ü": "U",
    "x": "x", "X": "X",  # x stays
}
DIACRITIC_PROB = 0.85


def strip_diacritics(text: str, prob: float = DIACRITIC_PROB) -> str:
    out = []
    for ch in text:
        if ch in DIACRITIC_MAP and random.random() < prob:
            out.append(DIACRITIC_MAP[ch])
        else:
            out.append(ch)
    return "".join(out)


def gen_diacritic_strip(words, sents, n=5000):
    pairs = []
    pool = random.sample(words, min(len(words), 3000)) + random.sample(sents, min(len(sents), 3000))
    random.shuffle(pool)
    for t in pool:
        if len(pairs) >= n:
            break
        n_t = strip_diacritics(t)
        if n_t != t:
            pairs.append({"clean": t, "noisy": n_t})
    return pairs


# ---------------------------------------------------------------------------
# 3. Capital İ / I confusion at word starts
# ---------------------------------------------------------------------------

def gen_capital_i(words, sents, n=3500):
    pool = []
    for w in words:
        if w[:1] == "İ":
            pool.append(("word", w))
    for s in sents:
        for token in s.split():
            tok_clean = token.strip(".,!?;:\"'()[]")
            if tok_clean[:1] == "İ" and len(tok_clean) <= 30:
                pool.append(("word", tok_clean))
    # also sentences that contain İ at word starts (we noise them)
    for s in sents:
        if any(w[:1] == "İ" for w in s.split()):
            pool.append(("sent", s))
    random.shuffle(pool)
    pairs = []
    for kind, txt in pool:
        if len(pairs) >= n:
            break
        # Variant 1: replace İ at word start with Latin I
        v1 = re.sub(r"\bİ", "I", txt)
        if v1 != txt:
            pairs.append({"clean": txt, "noisy": v1})
        # Variant 2: lowercase
        if random.random() < 0.4 and txt[:1].isupper():
            v2 = txt[0].lower() + txt[1:]
            if v2 != txt:
                pairs.append({"clean": txt, "noisy": v2})
        # Variant 3: NFD form (i + combining dot above)
        if random.random() < 0.25:
            v3 = txt.replace("İ", "İ")  # i + U+0307
            if v3 != txt:
                pairs.append({"clean": txt, "noisy": v3})
    return pairs[:n]


# ---------------------------------------------------------------------------
# 4. Keyboard-adjacent typos
# ---------------------------------------------------------------------------

# Simplified Azerbaijani QWERTY (lowercase only; we'll case-mirror)
ADJ = {
    "q": "üwa", "w": "qse", "e": "wsd", "r": "etf", "t": "ryg", "y": "tuh",
    "u": "yij", "i": "uok", "o": "ipl", "p": "oöü",
    "a": "qsz", "s": "adwxz", "d": "sfxce", "f": "dgcvr", "g": "fhbvt",
    "h": "gjnby", "j": "hkmnu", "k": "jlmi", "l": "kö",
    "z": "xs", "x": "zsdc", "c": "xdvf", "v": "cbgf", "b": "vgnh", "n": "bjmh", "m": "njk",
    "ə": "wae", "ş": "sxc", "ç": "vbn", "ğ": "qw", "ö": "öpl", "ü": "üpa", "ı": "uo",
}


def adjacent_typo(word: str, n_changes: int = 1) -> str:
    if len(word) < 2:
        return word
    chars = list(word)
    indices = list(range(len(chars)))
    random.shuffle(indices)
    changed = 0
    for i in indices:
        ch = chars[i].lower()
        if ch in ADJ:
            chars[i] = random.choice(ADJ[ch])
            changed += 1
            if changed >= n_changes:
                break
    return "".join(chars)


def gen_keyboard_adjacent(words, sents, n=3500):
    pairs = []
    pool = random.sample(words, min(len(words), n))
    for w in pool:
        if len(pairs) >= n:
            break
        nw = adjacent_typo(w, n_changes=random.randint(1, 2))
        if nw != w:
            pairs.append({"clean": w, "noisy": nw})
    # also sentence-level (one word inside the sentence)
    sent_count = min(n // 4, len(sents))
    for s in random.sample(sents, sent_count):
        if len(pairs) >= n + sent_count:
            break
        toks = s.split()
        if not toks:
            continue
        i = random.randrange(len(toks))
        nt = adjacent_typo(toks[i], n_changes=1)
        if nt != toks[i]:
            toks[i] = nt
            pairs.append({"clean": s, "noisy": " ".join(toks)})
    return pairs[:n]


# ---------------------------------------------------------------------------
# 5. Double-letter insertion
# ---------------------------------------------------------------------------

def gen_double_letters(words, sents, n=2500):
    pairs = []
    pool = random.sample(words, min(len(words), n))
    for w in pool:
        if len(pairs) >= n:
            break
        if len(w) < 2:
            continue
        i = random.randrange(len(w))
        nw = w[:i] + w[i] + w[i:]
        pairs.append({"clean": w, "noisy": nw})
    return pairs[:n]


# ---------------------------------------------------------------------------
# 6. Dropped-letter
# ---------------------------------------------------------------------------

def gen_dropped_letters(words, n=2500):
    pairs = []
    pool = random.sample(words, min(len(words), n))
    for w in pool:
        if len(pairs) >= n:
            break
        if len(w) < 3:
            continue
        i = random.randrange(1, len(w) - 1)  # don't drop first/last too often
        nw = w[:i] + w[i + 1:]
        if nw != w:
            pairs.append({"clean": w, "noisy": nw})
    return pairs[:n]


# ---------------------------------------------------------------------------
# 7. Letter-swap (adjacent transposition)
# ---------------------------------------------------------------------------

def gen_letter_swap(words, n=2500):
    pairs = []
    pool = random.sample(words, min(len(words), n))
    for w in pool:
        if len(pairs) >= n:
            break
        if len(w) < 3:
            continue
        i = random.randrange(len(w) - 1)
        chars = list(w)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
        nw = "".join(chars)
        if nw != w:
            pairs.append({"clean": w, "noisy": nw})
    return pairs[:n]


# ---------------------------------------------------------------------------
# 8. Case errors (lowercase capitalize, ALL CAPS)
# ---------------------------------------------------------------------------

def gen_case_errors(words, sents, n=2000):
    pairs = []
    # words starting with capital → lowercase noisy
    cap_words = [w for w in words if w[:1].isupper()]
    for w in random.sample(cap_words, min(len(cap_words), n // 2)):
        pairs.append({"clean": w, "noisy": w.lower()})
    # sentence-start capital → lowercase
    for s in random.sample(sents, min(len(sents), n // 2)):
        if s and s[0].isalpha() and s[0].isupper():
            pairs.append({"clean": s, "noisy": s[0].lower() + s[1:]})
    return pairs[:n]


# ---------------------------------------------------------------------------
# 9. Missing-space within sentence
# ---------------------------------------------------------------------------

def gen_missing_space(sents, n=2000):
    pairs = []
    pool = random.sample(sents, min(len(sents), n))
    for s in pool:
        if len(pairs) >= n:
            break
        toks = s.split()
        if len(toks) < 2:
            continue
        i = random.randrange(len(toks) - 1)
        if len(toks[i]) + len(toks[i + 1]) > 30:
            continue
        joined = toks[i] + toks[i + 1]
        ns = " ".join(toks[:i] + [joined] + toks[i + 2:])
        if ns != s:
            pairs.append({"clean": s, "noisy": ns})
    return pairs[:n]


# ---------------------------------------------------------------------------
# 10. Extra-space (split compound word)
# ---------------------------------------------------------------------------

def gen_extra_space(words, n=1200):
    pairs = []
    long_words = [w for w in words if len(w) >= 6]
    pool = random.sample(long_words, min(len(long_words), n))
    for w in pool:
        if len(pairs) >= n:
            break
        i = random.randrange(2, len(w) - 2)
        ns = w[:i] + " " + w[i:]
        pairs.append({"clean": w, "noisy": ns})
    return pairs[:n]


# ---------------------------------------------------------------------------
# 11. Russian / Cyrillic transliteration of Azerbaijani words
# ---------------------------------------------------------------------------

LATIN_TO_CYR = {
    "a": "а", "b": "б", "c": "ҹ", "ç": "ч", "d": "д", "e": "е", "ə": "ә",
    "f": "ф", "g": "ҝ", "ğ": "ғ", "h": "һ", "x": "х", "ı": "ы", "i": "и",
    "j": "ж", "k": "к", "q": "г", "l": "л", "m": "м", "n": "н", "o": "о",
    "ö": "ө", "p": "п", "r": "р", "s": "с", "ş": "ш", "t": "т", "u": "у",
    "ü": "ү", "v": "в", "y": "й", "z": "з",
}


def latin_to_cyrillic(word: str) -> str:
    out = []
    for ch in word:
        lo = ch.lower()
        repl = LATIN_TO_CYR.get(lo, ch)
        out.append(repl.upper() if ch.isupper() else repl)
    return "".join(out)


def gen_russian_translit(words, n=1500):
    pairs = []
    pool = random.sample(words, min(len(words), n))
    for w in pool:
        if len(pairs) >= n:
            break
        nw = latin_to_cyrillic(w)
        if nw != w and any(ord(c) > 0x400 for c in nw):
            pairs.append({"clean": w, "noisy": nw})
    return pairs[:n]


# ---------------------------------------------------------------------------
# 12. Turkish word forms used instead of AZ (small handcrafted map)
# ---------------------------------------------------------------------------

TURKISH_TO_AZ = {
    "merhaba": "salam",
    "evet": "bəli",
    "hayır": "xeyr",
    "günaydın": "sabahınız xeyir",
    "teşekkürler": "təşəkkürlər",
    "teşekkür": "təşəkkür",
    "nasıl": "necə",
    "nasılsın": "necəsən",
    "yapmak": "etmək",
    "yapacağım": "edəcəyəm",
    "olmaz": "olmaz",
    "var": "var",
    "yok": "yoxdur",
    "evde": "evdə",
    "okul": "məktəb",
    "öğrenci": "tələbə",
    "öğretmen": "müəllim",
    "araba": "maşın",
    "kitap": "kitab",
    "telefon": "telefon",
    "şehir": "şəhər",
    "köy": "kənd",
    "kardeş": "qardaş",
    "anne": "ana",
    "baba": "ata",
    "abla": "bacı",
    "yarın": "sabah",
    "bugün": "bu gün",
    "dün": "dünən",
    "şimdi": "indi",
    "lütfen": "zəhmət olmasa",
    "tamam": "yaxşı",
    "güzel": "gözəl",
    "büyük": "böyük",
    "küçük": "kiçik",
    "uzun": "uzun",
    "kısa": "qısa",
    "yeni": "yeni",
    "eski": "köhnə",
    "iyi": "yaxşı",
    "kötü": "pis",
    "sıcak": "isti",
    "soğuk": "soyuq",
    "mutlu": "xoşbəxt",
    "üzgün": "kədərli",
}


def gen_turkish_words(n=1500):
    pairs = []
    items = list(TURKISH_TO_AZ.items())
    # base pairs (word-level)
    for tk, az in items:
        for _ in range(3):
            pairs.append({"clean": az, "noisy": tk})
        # capitalized variant
        pairs.append({"clean": az.capitalize(), "noisy": tk.capitalize()})
    # short sentence templates
    templates = [
        ("{} dostum", "{} dostum"),
        ("{} necəsən", "{} necesin"),
        ("salam {}", "salam {}"),
        ("{} dedim", "{} dedim"),
    ]
    for tk, az in items:
        ct, nt = random.choice(templates)
        pairs.append({"clean": ct.format(az), "noisy": nt.format(tk)})
    random.shuffle(pairs)
    return pairs[:n]


# ---------------------------------------------------------------------------
# 13. English borrowings — preserve common loanwords as-is, plus map a few
# ---------------------------------------------------------------------------

ENG_TO_AZ = {
    "phone": "telefon",
    "computer": "kompüter",
    "manager": "menecer",
    "email": "e-poçt",
    "internet": "internet",
    "wifi": "wifi",
    "password": "parol",
    "online": "onlayn",
    "office": "ofis",
    "boss": "müdir",
    "meeting": "iclas",
    "school": "məktəb",
    "teacher": "müəllim",
    "student": "tələbə",
    "book": "kitab",
    "car": "maşın",
    "house": "ev",
    "friend": "dost",
    "brother": "qardaş",
    "sister": "bacı",
    "mother": "ana",
    "father": "ata",
    "morning": "səhər",
    "night": "gecə",
    "good": "yaxşı",
    "bad": "pis",
    "happy": "xoşbəxt",
    "money": "pul",
    "work": "iş",
    "home": "ev",
}


def gen_english_borrowings(n=1200):
    pairs = []
    for en, az in ENG_TO_AZ.items():
        for _ in range(4):
            pairs.append({"clean": az, "noisy": en})
        pairs.append({"clean": az.capitalize(), "noisy": en.capitalize()})
    # brand/model names that should be PRESERVED unchanged (input == output)
    brands = ["iPhone 13", "Samsung S21", "MacBook Pro", "Galaxy A52", "Xiaomi Mi 11",
              "iPad Air", "AirPods Pro", "Pixel 7", "Huawei P50", "Tesla Model S",
              "Microsoft Office", "Google Chrome", "Adobe Photoshop", "iOS 16", "Android 13"]
    for b in brands:
        for _ in range(4):
            pairs.append({"clean": b, "noisy": b})
    random.shuffle(pairs)
    return pairs[:n]


# ---------------------------------------------------------------------------
# 14. Digit ↔ letter confusion
# ---------------------------------------------------------------------------

DIGIT_TO_LETTER = {"0": "o", "1": "i", "3": "ə", "4": "a", "5": "s", "7": "t", "8": "b"}
LETTER_TO_DIGIT = {v: k for k, v in DIGIT_TO_LETTER.items()}
LETTER_TO_DIGIT.update({"l": "1", "I": "1", "O": "0", "S": "5", "B": "8"})


def digit_noise(word: str) -> str:
    out = []
    changed = False
    for ch in word:
        if not changed and ch.lower() in LETTER_TO_DIGIT and random.random() < 0.4:
            out.append(LETTER_TO_DIGIT[ch.lower()])
            changed = True
        else:
            out.append(ch)
    return "".join(out)


def gen_digit_letter(words, n=1000):
    pairs = []
    pool = random.sample(words, min(len(words), n))
    for w in pool:
        if len(pairs) >= n:
            break
        nw = digit_noise(w)
        if nw != w:
            pairs.append({"clean": w, "noisy": nw})
    return pairs[:n]


# ---------------------------------------------------------------------------
# 15. Smart quotes / dashes
# ---------------------------------------------------------------------------

QUOTE_MAP = {
    "“": "\"",  # "
    "”": "\"",  # "
    "‘": "'",   # '
    "’": "'",   # '
    "«": "\"",  # «
    "»": "\"",  # »
    "–": "-",   # –
    "—": "-",   # —
}


def gen_smart_quotes(sents, n=700):
    pairs = []
    pool = random.sample(sents, min(len(sents), n * 2))
    for s in pool:
        if len(pairs) >= n:
            break
        # Insert quoted segments if sentence has none
        ns = s
        if random.random() < 0.5:
            words = s.split()
            if len(words) > 3:
                i = random.randrange(len(words) - 2)
                j = i + random.randint(1, 2)
                # clean uses standard quotes, noisy uses smart quotes
                clean = " ".join(words[:i] + ['"' + " ".join(words[i:j+1]) + '"'] + words[j+1:])
                noisy = " ".join(words[:i] + ['“' + " ".join(words[i:j+1]) + '”'] + words[j+1:])
                pairs.append({"clean": clean, "noisy": noisy})
                continue
        # Default: swap any existing typographic char
        for cyr, plain in QUOTE_MAP.items():
            if cyr in s:
                pairs.append({"clean": s.replace(cyr, plain), "noisy": s})
                break
    return pairs[:n]


# ---------------------------------------------------------------------------
# 16. Slang / colloquial phonetic spelling
# ---------------------------------------------------------------------------

SLANG_PAIRS = [
    ("ona görə", "onagore"),
    ("nə var, nə yox", "nevar neyox"),
    ("necəsən", "necesen"),
    ("necə", "neca"),
    ("üçün", "ucun"),
    ("nə üçün", "ne ucun"),
    ("yox", "yoxh"),
    ("salam", "selam"),
    ("salam", "salamm"),
    ("nə xəbər", "ne xeber"),
    ("ata", "atta"),
    ("ana", "annaa"),
    ("çox sağ ol", "cox sagol"),
    ("təşəkkür edirəm", "tesekkur edirem"),
    ("yaxşı", "yaxsi"),
    ("yaxşı", "yaxshi"),
    ("yox", "yox"),
    ("bilmirəm", "bilmirem"),
    ("nə isə", "neyse"),
    ("istəyirəm", "isteyirem"),
    ("istəyirəm", "istiyirem"),
    ("oxşayır", "oxsayir"),
    ("görmüşəm", "gormusem"),
    ("getmişəm", "getmisem"),
    ("danışıram", "danisiram"),
    ("eləmişəm", "elemisem"),
    ("etmişəm", "etmisem"),
    ("yoxdu", "yoxdi"),
    ("ki", "kı"),
    ("biraz", "bi az"),
    ("indi", "indii"),
    ("indi", "ındi"),
    ("baxırsan", "baxrsan"),
    ("baxıram", "baxram"),
]


def gen_slang_colloquial(n=1500):
    pairs = []
    for clean, noisy in SLANG_PAIRS:
        for _ in range(8):
            pairs.append({"clean": clean, "noisy": noisy})
        pairs.append({"clean": clean.capitalize(), "noisy": noisy.capitalize()})
    random.shuffle(pairs)
    return pairs[:n]


# ---------------------------------------------------------------------------
# 17. Repeated letters (informal emphasis)
# ---------------------------------------------------------------------------

def gen_repeated_letters(words, n=700):
    pairs = []
    pool = random.sample(words, min(len(words), n))
    for w in pool:
        if len(pairs) >= n:
            break
        if len(w) < 2:
            continue
        i = random.randrange(len(w))
        repeats = random.choice([2, 3, 4])
        nw = w[:i + 1] + w[i] * repeats + w[i + 1:]
        pairs.append({"clean": w, "noisy": nw})
    return pairs[:n]


# ---------------------------------------------------------------------------
# 18. Proper nouns with İ / Ə / Ş written without diacritics
# ---------------------------------------------------------------------------

PROPER_NOUNS = [
    "İlham Əliyev", "Şamaxı", "İrəvan", "İçərişəhər", "İsmayıllı", "Şəki",
    "Qəbələ", "Şahdağ", "Lənkəran", "Naxçıvan", "Gəncə", "Sumqayıt",
    "Quba", "Şabran", "Balakən", "Cəlilabad", "Şirvan", "Mingəçevir",
    "Ağdam", "Füzuli", "Cəbrayıl", "Qubadlı", "Zəngilan", "Laçın",
    "Kəlbəcər", "Tovuz", "Şəmkir", "Göygöl", "Daşkəsən", "Goranboy",
    "İslam Əliyev", "Əli Həsənov", "Şəfa İsmayılova", "Sevda Əliyeva",
    "Cəlil Məmmədquluzadə", "Üzeyir Hacıbəyli", "Səməd Vurğun",
]


def gen_proper_nouns(n=1500):
    pairs = []
    for name in PROPER_NOUNS:
        for _ in range(4):
            pairs.append({"clean": name, "noisy": strip_diacritics(name, prob=1.0)})
        # Latin İ variants
        pairs.append({"clean": name, "noisy": re.sub(r"İ", "I", name)})
        # lowercase noise
        pairs.append({"clean": name, "noisy": name.lower()})
    random.shuffle(pairs)
    return pairs[:n]


# ---------------------------------------------------------------------------
# 19. Numbers and units spacing
# ---------------------------------------------------------------------------

UNIT_TEMPLATES = [
    ("{n} kilometr", ["{n}km", "{n}kilometr", "{n}-km"]),
    ("{n} metr", ["{n}m", "{n}metr"]),
    ("{n} kiloqram", ["{n}kg", "{n}kiloqram"]),
    ("{n} qram", ["{n}q", "{n}qram"]),
    ("{n} saat", ["{n}saat"]),
    ("{n} dəqiqə", ["{n}deqiqe", "{n}dəqiqə"]),
    ("{n} manat", ["{n}azn", "{n}man", "{n}manat", "{n}₼"]),
    ("{n} faiz", ["{n}%", "{n}faiz"]),
    ("{n} dərəcə", ["{n}°", "{n}derece"]),
    ("{n} il", ["{n}il", "{n}-il"]),
    ("{n} gün", ["{n}gun", "{n}gün"]),
]


def gen_numbers_units(n=1000):
    pairs = []
    for _ in range(n):
        clean_template, noisy_options = random.choice(UNIT_TEMPLATES)
        num = random.choice([1, 2, 5, 10, 15, 18, 20, 25, 30, 50, 100, 120, 200, 365, 1000, 1500, 2024, 2026])
        clean = clean_template.format(n=num)
        noisy = random.choice(noisy_options).format(n=num)
        if clean != noisy:
            pairs.append({"clean": clean, "noisy": noisy})
    return pairs[:n]


# ---------------------------------------------------------------------------
# 20. Single-diacritic-only typos (one char off)
# ---------------------------------------------------------------------------

SINGLE_DIACRITIC_MAP = {"ə": "e", "ş": "s", "ç": "c", "ğ": "g", "ı": "i", "ü": "u", "ö": "o", "İ": "I"}


def gen_single_diacritic(words, n=2500):
    pairs = []
    pool = random.sample(words, min(len(words), n * 3))
    for w in pool:
        if len(pairs) >= n:
            break
        # find positions of any diacritic
        positions = [i for i, c in enumerate(w) if c in SINGLE_DIACRITIC_MAP]
        if not positions:
            continue
        i = random.choice(positions)
        nw = w[:i] + SINGLE_DIACRITIC_MAP[w[i]] + w[i + 1:]
        pairs.append({"clean": w, "noisy": nw})
    return pairs[:n]


# ---------------------------------------------------------------------------
# Vocabulary filter — drop any pair that contains chars not in vocab.json
# ---------------------------------------------------------------------------

def filter_to_vocab(pairs, vocab):
    out = []
    dropped = 0
    for p in pairs:
        if all(c in vocab for c in p["clean"]) and all(c in vocab for c in p["noisy"]):
            out.append(p)
        else:
            dropped += 1
    return out, dropped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading source corpora...")
    words = load_words()
    sents = load_sentences()
    print(f"  {len(words):,} unique clean words")
    print(f"  {len(sents):,} unique clean sentences (8..110 chars)")

    print("\nLoading vocab for filtering...")
    vocab = json.load(open(PROJECT_ROOT / "data" / "pair" / "vocab.json"))
    print(f"  vocab size: {len(vocab)}")

    generators = [
        ("01_identity",                lambda: gen_identity(words, sents, 5000)),
        ("02_diacritic_strip",         lambda: gen_diacritic_strip(words, sents, 5000)),
        ("03_capital_i",               lambda: gen_capital_i(words, sents, 3500)),
        ("04_keyboard_adjacent",       lambda: gen_keyboard_adjacent(words, sents, 3500)),
        ("05_double_letters",          lambda: gen_double_letters(words, sents, 2500)),
        ("06_dropped_letters",         lambda: gen_dropped_letters(words, 2500)),
        ("07_letter_swap",             lambda: gen_letter_swap(words, 2500)),
        ("08_case_errors",             lambda: gen_case_errors(words, sents, 2000)),
        ("09_missing_space",           lambda: gen_missing_space(sents, 2000)),
        ("10_extra_space",             lambda: gen_extra_space(words, 1200)),
        ("11_russian_translit",        lambda: gen_russian_translit(words, 1500)),
        ("12_turkish_words",           lambda: gen_turkish_words(1500)),
        ("13_english_borrowings",      lambda: gen_english_borrowings(1200)),
        ("14_digit_letter_confusion",  lambda: gen_digit_letter(words, 1000)),
        ("15_smart_quotes",            lambda: gen_smart_quotes(sents, 700)),
        ("16_slang_colloquial",        lambda: gen_slang_colloquial(1500)),
        ("17_repeated_letters",        lambda: gen_repeated_letters(words, 700)),
        ("18_proper_nouns",            lambda: gen_proper_nouns(1500)),
        ("19_numbers_units",           lambda: gen_numbers_units(1000)),
        ("20_single_diacritic",        lambda: gen_single_diacritic(words, 2500)),
    ]

    all_pairs = []
    summary = []
    for name, fn in generators:
        pairs = fn()
        pairs, dropped = filter_to_vocab(pairs, vocab)
        out_path = OUT_DIR / f"{name}.json"
        json.dump(pairs, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        all_pairs.extend(pairs)
        summary.append((name, len(pairs), dropped))
        print(f"  {name:<35} -> {len(pairs):>6,} pairs (dropped OOV: {dropped})")

    random.shuffle(all_pairs)
    combined = OUT_DIR / "all_use_cases.json"
    json.dump(all_pairs, open(combined, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"\nCombined: {len(all_pairs):,} pairs -> {combined}")

    # Write a short summary file (JSON, not markdown — easier to parse)
    summary_path = OUT_DIR / "_summary.json"
    json.dump(
        {
            "categories": [
                {"name": n, "count": c, "dropped_oov": d}
                for (n, c, d) in summary
            ],
            "total": len(all_pairs),
        },
        open(summary_path, "w", encoding="utf-8"),
        indent=2,
        ensure_ascii=False,
    )
    print(f"Summary: {summary_path}")

    # Print 1 sample per category
    print("\n=== Samples ===")
    for n, _, _ in summary:
        data = json.load(open(OUT_DIR / f"{n}.json"))
        if data:
            ex = random.choice(data)
            c = ex["clean"][:60]
            ns = ex["noisy"][:60]
            print(f"  {n:<32} clean: {c!r:<55}  noisy: {ns!r}")


if __name__ == "__main__":
    main()
