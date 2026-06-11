"""
Compare the old best checkpoint to the freshly trained one.

Runs evaluate_model.py-style metrics on both checkpoints over the same
sampled test slices and prints a side-by-side delta.
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
import sys
from pathlib import Path

import torch
from jiwer import cer
from rapidfuzz.distance import Levenshtein
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tokenizer import Tokenizer
from train_attention_mps import Seq2SeqAttentionModel, load_pairs_from_path


def get_device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def build_model(ckpt_path, vocab_size, pad_idx, device, default_max_len=128):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck.get("config") or {}
    model = Seq2SeqAttentionModel(
        vocab_size=vocab_size,
        embedding_dim=cfg.get("embedding_dim", 256),
        hidden_dim=cfg.get("hidden_dim", 384),
        num_layers=cfg.get("num_layers", 1),
        dropout=cfg.get("dropout", 0.1),
        n_heads=cfg.get("n_heads", 4),
        pad_idx=pad_idx,
    ).to(device)
    # The old checkpoint may have vocab_size=484, new one 485.
    # Pad old state_dict rows to match local model if needed.
    sd = ck["model_state_dict"]
    fixed = {}
    for k, v in sd.items():
        if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] != vocab_size:
            tgt_first = model.state_dict()[k].shape[0]
            if v.shape[0] < tgt_first:
                if v.dim() == 1:
                    pad = torch.zeros(tgt_first - v.shape[0], dtype=v.dtype)
                    v = torch.cat([v, pad], dim=0)
                else:
                    pad = torch.zeros(tgt_first - v.shape[0], v.shape[1], dtype=v.dtype)
                    v = torch.cat([v, pad], dim=0)
        fixed[k] = v
    model.load_state_dict(fixed)
    model.eval()
    return model, ck.get("epoch"), ck.get("val_loss"), cfg.get("max_length", default_max_len)


def token_id(tok, names):
    for n in names:
        try:
            v = tok.token_to_id(n)
            if v is not None:
                return v
        except Exception:
            pass
    return None


def decode_ids(tok, ids, pad_id, bos_id, eos_id):
    out = []
    for i in ids:
        if i == eos_id:
            break
        if i in (pad_id, bos_id):
            continue
        out.append(i)
    try:
        return tok.decode(out)
    except Exception:
        return "".join(tok.id_to_token(i) for i in out)


@torch.inference_mode()
def batch_predict(model, tok, texts, device, max_length, batch_size=192):
    pad = tok.vocab["<PAD>"]
    bos = tok.vocab["<BOS>"]
    eos = tok.vocab["<EOS>"]
    out = []
    for s in tqdm(range(0, len(texts), batch_size), desc="decode", ncols=80, leave=False):
        chunk = texts[s:s + batch_size]
        noisy = []
        for t in chunk:
            ids = tok.encode(t)[:max_length]
            ids = ids + [pad] * (max_length - len(ids))
            noisy.append(ids)
        n = torch.tensor(noisy, dtype=torch.long, device=device)
        c = torch.full((len(chunk), max_length), pad, dtype=torch.long, device=device)
        c[:, 0] = bos
        logits, _ = model(n, c, teacher_forcing_ratio=0.0, return_attention=False)
        pred = logits.argmax(dim=-1).tolist()
        for row in pred:
            out.append(decode_ids(tok, row[1:], pad, bos, eos))
    return out


def eval_metrics(noisy, clean, preds):
    n = len(clean)
    exact = sum(1 for c, p in zip(clean, preds) if c == p) / n
    cer_val = cer(clean, [p if p else " " for p in preds])
    regressions = sum(
        1 for nn, cc, pp in zip(noisy, clean, preds)
        if pp != cc and Levenshtein.distance(pp, cc) >= Levenshtein.distance(nn, cc)
    )
    return {"exact_match": exact, "cer": cer_val, "regressions": regressions, "n": n}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--old", default="checkpoints/best_attention_model.pt.bak")
    p.add_argument("--new", default="checkpoints/best_attention_model.pt")
    p.add_argument("--vocab", default="data/pair/vocab.json")
    p.add_argument("--test-dir", default="data/pair/test")
    p.add_argument("--sample-limit", type=int, default=2000)
    p.add_argument("--max-length", type=int, default=128)
    args = p.parse_args()

    device = get_device()
    print("device:", device)

    tok = Tokenizer(max_length=args.max_length)
    tok.load_vocab(args.vocab)
    vocab_size = len(tok.vocab)
    pad = tok.vocab["<PAD>"]

    # Both models loaded with vocab_size=485 (pad the old one)
    old_model, old_ep, old_loss, _ = build_model(args.old, vocab_size, pad, device)
    new_model, new_ep, new_loss, _ = build_model(args.new, vocab_size, pad, device)
    print(f"old: epoch={old_ep}  val_loss={old_loss:.4f}")
    print(f"new: epoch={new_ep}  val_loss={new_loss:.4f}")

    files = sorted(Path(args.test_dir).glob("*.json"))
    for fp in files:
        items, _ = load_pairs_from_path(fp)
        items = items[:args.sample_limit]
        noisy = [it["noisy"] for it in items]
        clean = [it["clean"] for it in items]

        # Old model needs max_length=64 to behave consistently with training
        old_preds = batch_predict(old_model, tok, noisy, device, max_length=64)
        new_preds = batch_predict(new_model, tok, noisy, device, max_length=args.max_length)

        m_old = eval_metrics(noisy, clean, old_preds)
        m_new = eval_metrics(noisy, clean, new_preds)
        print(f"\n### {fp.name}  (n={len(items)})")
        print(f"  metric           OLD            NEW            delta")
        print(f"  exact-match      {m_old['exact_match']:.4f}         {m_new['exact_match']:.4f}         {m_new['exact_match']-m_old['exact_match']:+.4f}")
        print(f"  CER              {m_old['cer']:.4f}         {m_new['cer']:.4f}         {m_new['cer']-m_old['cer']:+.4f}")
        print(f"  regressions      {m_old['regressions']}/{m_old['n']}     {m_new['regressions']}/{m_new['n']}")


if __name__ == "__main__":
    main()
