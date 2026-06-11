"""
Add 'İ' (U+0130, Azerbaijani capital dotted I) to the vocabulary and extend
the saved model's embedding/output matrices by one row.

The new row is initialized from the existing 'i' embedding so it starts in a
sensible region of weight space.

Files updated:
  - data/pair/vocab.json                 (vocab size 484 -> 485)
  - src/tokenizer.py                     (alphabet typo fix, so future rebuilds work)
  - checkpoints/best_attention_model_extended.pt  (new checkpoint with widened weights)
"""

import json
import shutil
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
VOCAB_PATH = PROJECT_ROOT / "data" / "pair" / "vocab.json"
CKPT_IN = PROJECT_ROOT / "checkpoints" / "best_attention_model.pt"
CKPT_OUT = PROJECT_ROOT / "checkpoints" / "best_attention_model_extended.pt"
TOKENIZER_PATH = PROJECT_ROOT / "src" / "tokenizer.py"


def extend_vocab():
    vocab = json.load(open(VOCAB_PATH, "r", encoding="utf-8"))
    if "İ" in vocab:
        print("İ already in vocab, nothing to do for vocab.")
        return len(vocab), False
    new_id = len(vocab)
    vocab["İ"] = new_id
    json.dump(vocab, open(VOCAB_PATH, "w", encoding="utf-8"),
              indent=4, ensure_ascii=False)
    print(f"Added 'İ' at id {new_id}. Vocab size: {len(vocab)}")
    return len(vocab), True


def extend_checkpoint(new_vocab_size: int, source_token_for_init: str = "i"):
    """Append a row to embedding/output matrices for the new token.

    Initialize the new row from `source_token_for_init`'s vector so it starts
    near a sensible region (capital İ and lowercase i are semantically close).
    """
    # Need 'i' index from current vocab to use for initialization
    vocab = json.load(open(VOCAB_PATH, "r", encoding="utf-8"))
    i_id = vocab[source_token_for_init]
    new_id = vocab["İ"]

    ckpt = torch.load(CKPT_IN, map_location="cpu", weights_only=False)
    msd = ckpt["model_state_dict"]

    # Detect things that need extending: anything with first dim == old vocab size
    old_size = new_vocab_size - 1
    targets = []
    for k, v in msd.items():
        if not torch.is_tensor(v):
            continue
        if v.dim() >= 1 and v.shape[0] == old_size:
            targets.append((k, "row"))
        # fc_out bias has shape [vocab_size]
    print(f"Tensors to extend (matching first dim == {old_size}):")
    for k, _ in targets:
        print(f"  {k:<60} shape={tuple(msd[k].shape)}")

    for k, mode in targets:
        v = msd[k]
        if v.dim() == 1:
            # bias-like, shape [vocab_size]
            new_v = torch.zeros(v.shape[0] + 1, dtype=v.dtype)
            new_v[:old_size] = v
            new_v[new_id] = v[i_id]
            msd[k] = new_v
        elif v.dim() == 2:
            new_v = torch.zeros(v.shape[0] + 1, v.shape[1], dtype=v.dtype)
            new_v[:old_size] = v
            new_v[new_id] = v[i_id]
            msd[k] = new_v
        else:
            print(f"  WARN: skipped {k} (dim={v.dim()})")

    # Optimizer state has tensors that are tied to parameter shapes. We must
    # invalidate it (saving anyway breaks shape consistency).
    # Drop optimizer/scheduler state — training script will start fresh
    # optimizer state but resume from epoch + best_val_loss.
    print("Dropping optimizer/scheduler state (shapes changed).")
    ckpt.pop("optimizer_state_dict", None)
    ckpt.pop("scheduler_state_dict", None)
    ckpt["vocab_size"] = new_vocab_size

    torch.save(ckpt, CKPT_OUT)
    print(f"Wrote extended checkpoint to {CKPT_OUT}")


def fix_tokenizer_alphabet():
    """The alphabet string in tokenizer.py has a typo (two `I`s, missing `İ`).
    Fix it so future vocab rebuilds include İ.
    """
    src = TOKENIZER_PATH.read_text(encoding="utf-8")
    # Original line has: "ABCÇDEƏFGĞHXIIJKQLMNOÖPRSŞTUÜVYZ"
    bad = '"ABCÇDEƏFGĞHXIIJKQLMNOÖPRSŞTUÜVYZ"'
    good = '"ABCÇDEƏFGĞHXIİJKQLMNOÖPRSŞTUÜVYZ"'
    if bad not in src:
        print("Tokenizer alphabet already fixed (or pattern not found).")
        return
    TOKENIZER_PATH.write_text(src.replace(bad, good), encoding="utf-8")
    print("Fixed tokenizer.py alphabet (now includes İ).")


def main():
    print("=" * 60)
    print("Backing up checkpoint and vocab ...")
    shutil.copy(CKPT_IN, str(CKPT_IN) + ".bak")
    shutil.copy(VOCAB_PATH, str(VOCAB_PATH) + ".bak")

    new_size, changed = extend_vocab()
    if changed:
        extend_checkpoint(new_size)
    fix_tokenizer_alphabet()
    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
