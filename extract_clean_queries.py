"""
Extract clean Azerbaijani queries from all in-domain sources we have:

  - search_dlp/benchmark_dataset_dlp.zip      (~88 CSVs, ~50 queries each)
  - data/benchmark_dataset_dlp_v2/*.csv       (~97 CSVs, ~150 queries each)
  - data/benchmark_full.csv                   (14.4k rows)
  - data/reformulations.json                  (reformulated alternatives)

Outputs:
  data/llm_input/clean_queries.jsonl
    one line per query: {"index": N, "query": "<clean azerbaijani query>"}
"""

import csv
import io
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
NAIC_ROOT = PROJECT_ROOT.parent

SOURCES = {
    "search_dlp_zip": NAIC_ROOT / "search_dlp" / "benchmark_dataset_dlp.zip",
    "v2_dir":         NAIC_ROOT / "data" / "benchmark_dataset_dlp_v2",
    "benchmark_full": NAIC_ROOT / "data" / "benchmark_full.csv",
    "reformulations": NAIC_ROOT / "data" / "reformulations.json",
    "mining_queries": NAIC_ROOT / "data" / "mining_queries.json",
}

OUT_PATH = PROJECT_ROOT / "data" / "llm_input" / "clean_queries.jsonl"


def is_good_query(q: str) -> bool:
    """Basic sanity filter for a clean query candidate."""
    if not isinstance(q, str):
        return False
    q = q.strip()
    if not q:
        return False
    if len(q) < 2 or len(q) > 120:
        return False
    # must contain at least one Azerbaijani letter or basic Latin letter
    if not re.search(r"[a-zA-ZəıöüçşğƏIİÖÜÇŞĞ]", q):
        return False
    # reject queries that are all caps + numbers (often HS code rows)
    if re.fullmatch(r"[\d;\s,.\"'-]+", q):
        return False
    return True


def from_csv_text(text: str, query_col: str = "query"):
    """Yield query strings from CSV text content."""
    rdr = csv.DictReader(io.StringIO(text))
    for row in rdr:
        q = row.get(query_col)
        if is_good_query(q):
            yield q.strip()


def harvest_zip(path: Path):
    out = []
    if not path.exists():
        return out
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if name.startswith("__MACOSX/"):
                continue
            if not name.endswith(".csv"):
                continue
            try:
                with zf.open(name) as f:
                    text = f.read().decode("utf-8", errors="replace")
                out.extend(from_csv_text(text))
            except Exception as e:
                print(f"  WARN: failed to read {name}: {e}")
    return out


def harvest_csv_dir(d: Path):
    out = []
    if not d.exists():
        return out
    for p in sorted(d.glob("*.csv")):
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            out.extend(from_csv_text(text))
        except Exception as e:
            print(f"  WARN: failed to read {p.name}: {e}")
    return out


def harvest_csv_file(p: Path, query_col="query"):
    if not p.exists():
        return []
    out = []
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        out = list(from_csv_text(f.read(), query_col=query_col))
    return out


def harvest_reformulations(p: Path):
    if not p.exists():
        return []
    out = []
    try:
        data = json.load(open(p, "r", encoding="utf-8"))
    except Exception as e:
        print(f"  WARN: reformulations parse failed: {e}")
        return out
    for item in data:
        if isinstance(item, dict):
            q = item.get("query")
            if is_good_query(q):
                out.append(q.strip())
            sample = item.get("sample")
            if isinstance(sample, dict):
                for v in sample.values():
                    if isinstance(v, str) and is_good_query(v):
                        out.append(v.strip())
                    elif isinstance(v, list):
                        for s in v:
                            if isinstance(s, str) and is_good_query(s):
                                out.append(s.strip())
    return out


def harvest_mining(p: Path):
    if not p.exists():
        return []
    try:
        data = json.load(open(p, "r", encoding="utf-8"))
    except Exception:
        return []
    out = []
    if isinstance(data, list):
        for x in data:
            if isinstance(x, str):
                if is_good_query(x):
                    out.append(x.strip())
            elif isinstance(x, dict):
                q = x.get("query")
                if is_good_query(q):
                    out.append(q.strip())
    return out


def main():
    print("Extracting clean Azerbaijani queries from in-domain sources...\n")

    buckets = {}
    buckets["search_dlp_zip"] = harvest_zip(SOURCES["search_dlp_zip"])
    print(f"  search_dlp_zip:      {len(buckets['search_dlp_zip']):,}")

    buckets["v2_dir"] = harvest_csv_dir(SOURCES["v2_dir"])
    print(f"  v2_dir:              {len(buckets['v2_dir']):,}")

    buckets["benchmark_full"] = harvest_csv_file(SOURCES["benchmark_full"])
    print(f"  benchmark_full:      {len(buckets['benchmark_full']):,}")

    buckets["reformulations"] = harvest_reformulations(SOURCES["reformulations"])
    print(f"  reformulations:      {len(buckets['reformulations']):,}")

    buckets["mining_queries"] = harvest_mining(SOURCES["mining_queries"])
    print(f"  mining_queries:      {len(buckets['mining_queries']):,}")

    all_queries = []
    for v in buckets.values():
        all_queries.extend(v)
    print(f"\nTotal (raw):           {len(all_queries):,}")

    # dedupe (case-insensitive, whitespace-normalized)
    seen = set()
    uniq = []
    for q in all_queries:
        key = " ".join(q.lower().split())
        if key not in seen:
            seen.add(key)
            uniq.append(q.strip())

    print(f"Total (unique):        {len(uniq):,}")

    # quick distribution
    lens = [len(q) for q in uniq]
    print(f"Query length: min={min(lens)} max={max(lens)} mean={sum(lens)/len(lens):.1f}")

    # write JSONL
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for i, q in enumerate(uniq):
            f.write(json.dumps({"index": i, "query": q}, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(uniq):,} queries to {OUT_PATH}")

    # print 10 random samples
    import random
    random.seed(0)
    print("\nSamples:")
    for q in random.sample(uniq, min(15, len(uniq))):
        print(f"  {q!r}")


if __name__ == "__main__":
    main()
