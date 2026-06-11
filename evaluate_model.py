"""
Evaluate the best saved attention checkpoint on the test sets.

Reports:
  - Exact-match accuracy (word-level and sentence-level)
  - Character Error Rate (CER) via jiwer
  - Identity-baseline (what % would already be right if we returned the input)
  - Breakdown by perturbation difficulty (Levenshtein distance noisy->clean)
  - Per-test-file worst cases dumped to JSON for inspection
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm
from jiwer import cer
from rapidfuzz.distance import Levenshtein

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tokenizer import Tokenizer
from train_attention_mps import Seq2SeqAttentionModel, load_pairs_from_path


def get_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def token_id(tokenizer, names):
    for n in names:
        try:
            v = tokenizer.token_to_id(n)
            if v is not None:
                return v
        except Exception:
            pass
    return None


def decode_ids(tokenizer, ids, pad_id, bos_id, eos_id):
    cleaned = []
    for idx in ids:
        if idx == eos_id:
            break
        if idx in (pad_id, bos_id):
            continue
        cleaned.append(idx)
    try:
        return tokenizer.decode(cleaned)
    except Exception:
        return "".join(tokenizer.id_to_token(i) for i in cleaned)


@torch.inference_mode()
def batch_greedy_correct(model, tokenizer, texts, device, max_length=64, batch_size=128):
    """Run the model on a list of strings, return list of corrected strings."""
    pad_id = tokenizer.token_to_id("<PAD>")
    bos_id = token_id(tokenizer, ["<BOS>", "<SOS>", "<START>"])
    eos_id = token_id(tokenizer, ["<EOS>", "</s>", "<END>"])

    out = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Decoding", ncols=100):
        chunk = texts[start:start + batch_size]
        noisy = []
        for t in chunk:
            enc = tokenizer.encode(t)
            ids = enc.ids if hasattr(enc, "ids") else enc
            ids = ids[:max_length]
            ids = ids + [pad_id] * (max_length - len(ids))
            noisy.append(ids)
        noisy_tensor = torch.tensor(noisy, dtype=torch.long, device=device)
        # dummy target: only needs trg_len; we run with no teacher forcing
        clean_dummy = torch.full((len(chunk), max_length), pad_id, dtype=torch.long, device=device)
        clean_dummy[:, 0] = bos_id if bos_id is not None else 0

        outputs, _ = model(noisy_tensor, clean_dummy, teacher_forcing_ratio=0.0, return_attention=False)
        pred = outputs.argmax(dim=-1).tolist()
        for row in pred:
            out.append(decode_ids(tokenizer, row[1:], pad_id, bos_id, eos_id))
    return out


def edit_distance_bucket(noisy, clean):
    d = Levenshtein.distance(noisy, clean)
    if d == 0:
        return "0 (identical)"
    if d == 1:
        return "1"
    if d == 2:
        return "2"
    if d <= 4:
        return "3-4"
    return "5+"


def evaluate_file(path: Path, model, tokenizer, device, max_length=64, batch_size=128, sample_limit=None):
    items, skipped = load_pairs_from_path(path)
    if sample_limit:
        items = items[:sample_limit]
    print(f"\n=== {path.name}: {len(items):,} pairs (skipped {skipped}) ===")

    noisy = [it["noisy"] for it in items]
    clean = [it["clean"] for it in items]

    preds = batch_greedy_correct(model, tokenizer, noisy, device, max_length=max_length, batch_size=batch_size)

    # metrics
    correct = sum(1 for p, c in zip(preds, clean) if p == c)
    accuracy = correct / len(items)

    identity_correct = sum(1 for n, c in zip(noisy, clean) if n == c)
    identity_accuracy = identity_correct / len(items)

    # CER (only on items where clean is non-empty)
    pred_for_cer = [p if p else " " for p in preds]
    cer_value = cer(clean, pred_for_cer)
    identity_cer = cer(clean, [n if n else " " for n in noisy])

    # bucketed accuracy by input difficulty
    buckets = {"0 (identical)": [0, 0], "1": [0, 0], "2": [0, 0], "3-4": [0, 0], "5+": [0, 0]}
    for n, c, p in zip(noisy, clean, preds):
        b = edit_distance_bucket(n, c)
        buckets[b][1] += 1
        if p == c:
            buckets[b][0] += 1

    # Find worst cases: pred differs from clean and pred is even further than noisy
    worst = []
    for n, c, p in zip(noisy, clean, preds):
        if p == c:
            continue
        d_pred = Levenshtein.distance(p, c)
        d_noisy = Levenshtein.distance(n, c)
        if d_pred >= d_noisy:
            worst.append({
                "noisy": n,
                "clean": c,
                "pred": p,
                "noisy_dist": d_noisy,
                "pred_dist": d_pred,
                "regression": d_pred - d_noisy,
            })
    worst.sort(key=lambda r: (-r["regression"], -r["pred_dist"]))

    # categorize a sample of mistakes
    mistakes = [(n, c, p) for n, c, p in zip(noisy, clean, preds) if p != c]

    print(f"  Exact-match accuracy:   {accuracy:.4f}  ({correct}/{len(items)})")
    print(f"  Identity baseline:      {identity_accuracy:.4f}  (% where input already equals target)")
    print(f"  Character Error Rate:   {cer_value:.4f}")
    print(f"  Identity CER baseline:  {identity_cer:.4f}")
    print(f"  Mistakes total:         {len(mistakes):,}")
    print(f"  Regressions (pred worse than input): {len(worst):,}")
    print(f"  Accuracy by noise level:")
    for b, (ok, tot) in buckets.items():
        if tot == 0:
            continue
        print(f"    {b:>15}: {ok/tot:.4f}  ({ok}/{tot})")

    return {
        "file": path.name,
        "n": len(items),
        "accuracy": accuracy,
        "identity_baseline_accuracy": identity_accuracy,
        "cer": cer_value,
        "identity_cer": identity_cer,
        "buckets": {k: {"correct": v[0], "total": v[1]} for k, v in buckets.items()},
        "mistake_examples": [{"noisy": n, "clean": c, "pred": p} for n, c, p in mistakes[:30]],
        "regressions_top": worst[:30],
        "regression_count": len(worst),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_attention_model.pt")
    parser.add_argument("--vocab", default="data/pair/vocab.json")
    parser.add_argument("--test-dir", default="data/pair/test")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sample-limit", type=int, default=None,
                        help="If set, only evaluate the first N items of each file (for speed).")
    parser.add_argument("--report", default="eval_report.json")
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()

    device = get_device(args.force_cpu)
    print(f"Device: {device}")

    tokenizer = Tokenizer(max_length=args.max_length)
    tokenizer.load_vocab(args.vocab)
    vocab_size = len(tokenizer.vocab)
    pad_idx = tokenizer.token_to_id("<PAD>")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config") or {}
    embedding_dim = cfg.get("embedding_dim", 256)
    hidden_dim = cfg.get("hidden_dim", 384)
    num_layers = cfg.get("num_layers", 1)
    dropout = cfg.get("dropout", 0.1)
    n_heads = cfg.get("n_heads", 4)
    print(f"Loaded checkpoint epoch={ckpt.get('epoch')} val_loss={ckpt.get('val_loss'):.4f}")
    print(f"Arch: emb={embedding_dim} hidden={hidden_dim} layers={num_layers} heads={n_heads} dropout={dropout}")
    print(f"Vocab in ckpt={ckpt.get('vocab_size')}  local vocab={vocab_size}")

    model = Seq2SeqAttentionModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        n_heads=n_heads,
        pad_idx=pad_idx,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_dir = Path(args.test_dir)
    files = sorted([p for p in test_dir.iterdir() if p.suffix in {".json", ".jsonl"} and p.is_file()])

    report = {"checkpoint": args.checkpoint, "files": []}
    for fp in files:
        r = evaluate_file(fp, model, tokenizer, device, max_length=args.max_length,
                          batch_size=args.batch_size, sample_limit=args.sample_limit)
        report["files"].append(r)

    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote report to {args.report}")


if __name__ == "__main__":
    main()
