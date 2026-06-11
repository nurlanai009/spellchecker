"""
Train MPS-optimized Seq2Seq spellchecker with bidirectional LSTM encoder,
multi-head attention decoder, and layer normalization.

This is an optimized version of your first model, combined with the training
pipeline from your second script.

Expected project layout:
    spellchecker/
        train_attention_mps.py
        src/
            tokenizer.py
        data/
            pair/
                train/
                test/
                vocab.json
        checkpoints/
"""

import os

# Helps MPS fall back to CPU for unsupported ops instead of crashing.
# For best effect, this should be set before importing torch.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
import math
import random
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# Make src/tokenizer.py importable
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tokenizer import Tokenizer


# ----------------------------
# Reproducibility
# ----------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------
# Device helpers
# ----------------------------

def get_best_device(force_cpu: bool = False):
    if force_cpu:
        return torch.device("cpu")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def autocast_context(device, use_amp: bool):
    """
    AMP is disabled by default because LSTM + MPS mixed precision can be unstable
    depending on PyTorch version.

    Enable with:
        python train_attention_mps.py --amp
    """
    if not use_amp:
        return nullcontext()

    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    if device.type == "mps":
        return torch.autocast(device_type="mps", dtype=torch.float16)

    return nullcontext()


# ----------------------------
# Data loading
# ----------------------------

def _iter_pairs_from_file(path: Path):
    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    yield json.loads(line)
                except Exception:
                    continue

    elif path.suffix == ".json":
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return

        if isinstance(data, list):
            for item in data:
                yield item
        else:
            yield data


def load_pairs_from_path(path: Path):
    items = []
    skipped = 0

    if path.is_dir():
        files = sorted(
            p for p in path.iterdir()
            if p.is_file() and p.suffix in {".json", ".jsonl"}
        )
    else:
        files = [path]

    for fp in files:
        for item in _iter_pairs_from_file(fp):
            if not isinstance(item, dict):
                skipped += 1
                continue

            clean = item.get("clean") or item.get("original") or item.get("text")
            noisy = item.get("noisy")

            if clean is None or noisy is None:
                skipped += 1
                continue

            items.append({"clean": clean, "noisy": noisy})

    return items, skipped


class SpellCheckerDataset(Dataset):
    def __init__(self, data_items, tokenizer: Tokenizer, max_length: int = 64):
        self.data = data_items
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.token_to_id("<PAD>")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        clean = item.get("clean") or item.get("original") or item.get("text")
        noisy = item.get("noisy")

        if clean is None or noisy is None:
            raise KeyError("Each item must contain clean/original/text and noisy fields.")

        clean_enc = self.tokenizer.encode(clean)
        noisy_enc = self.tokenizer.encode(noisy)

        clean_ids_raw = clean_enc.ids if hasattr(clean_enc, "ids") else clean_enc
        noisy_ids_raw = noisy_enc.ids if hasattr(noisy_enc, "ids") else noisy_enc

        clean_ids = clean_ids_raw[:self.max_length]
        noisy_ids = noisy_ids_raw[:self.max_length]

        clean_ids = clean_ids + [self.pad_id] * (self.max_length - len(clean_ids))
        noisy_ids = noisy_ids + [self.pad_id] * (self.max_length - len(noisy_ids))

        return {
            "clean_ids": torch.tensor(clean_ids, dtype=torch.long),
            "noisy_ids": torch.tensor(noisy_ids, dtype=torch.long),
            "clean_length": min(len(clean_ids_raw), self.max_length),
            "noisy_length": min(len(noisy_ids_raw), self.max_length),
        }


def make_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
):
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": False,
    }

    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4

    return DataLoader(**kwargs)


# ----------------------------
# Model
# ----------------------------

class Encoder(nn.Module):
    """
    Bidirectional LSTM encoder.

    If decoder hidden_dim is 384, this encoder uses hidden_dim // 2 per direction,
    producing encoder outputs with dimension 384.
    """

    def __init__(
        self,
        input_dim,
        embedding_dim,
        enc_hidden_dim,
        num_layers,
        dropout,
        pad_idx=0,
    ):
        super().__init__()

        self.enc_hidden_dim = enc_hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(
            input_dim,
            embedding_dim,
            padding_idx=pad_idx,
        )

        self.embed_norm = nn.LayerNorm(embedding_dim)

        self.rnn = nn.LSTM(
            embedding_dim,
            enc_hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output_norm = nn.LayerNorm(enc_hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        batch_size = src.size(0)

        embedded = self.embedding(src)
        embedded = self.embed_norm(embedded)
        embedded = self.dropout(embedded)

        outputs, (hidden, cell) = self.rnn(embedded)
        outputs = self.output_norm(outputs)

        hidden = hidden.reshape(self.num_layers, 2, batch_size, self.enc_hidden_dim)
        hidden = torch.cat((hidden[:, 0], hidden[:, 1]), dim=-1)

        cell = cell.reshape(self.num_layers, 2, batch_size, self.enc_hidden_dim)
        cell = torch.cat((cell[:, 0], cell[:, 1]), dim=-1)

        return outputs, hidden, cell


class MultiHeadAttention(nn.Module):
    def __init__(self, enc_output_dim, dec_hidden_dim, n_heads, dropout=0.1):
        super().__init__()

        if dec_hidden_dim % n_heads != 0:
            raise ValueError("dec_hidden_dim must be divisible by n_heads")

        self.enc_output_dim = enc_output_dim
        self.dec_hidden_dim = dec_hidden_dim
        self.n_heads = n_heads
        self.head_dim = dec_hidden_dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.query_proj = nn.Linear(dec_hidden_dim, dec_hidden_dim)
        self.key_proj = nn.Linear(enc_output_dim, dec_hidden_dim)
        self.value_proj = nn.Linear(enc_output_dim, dec_hidden_dim)
        self.output_proj = nn.Linear(dec_hidden_dim, dec_hidden_dim)

        self.norm_q = nn.LayerNorm(dec_hidden_dim)
        self.norm_k = nn.LayerNorm(dec_hidden_dim)
        self.norm_v = nn.LayerNorm(dec_hidden_dim)
        self.norm_out = nn.LayerNorm(dec_hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value, mask=None, need_weights=False):
        """
        query:
            [batch_size, dec_hidden_dim]
        key_value:
            [batch_size, src_len, enc_output_dim]
        mask:
            [batch_size, 1, src_len]
        """
        batch_size = key_value.size(0)
        src_len = key_value.size(1)

        query = query.unsqueeze(1)
        query_len = 1

        q = self.norm_q(self.query_proj(query))
        k = self.norm_k(self.key_proj(key_value))
        v = self.norm_v(self.value_proj(key_value))

        q = q.reshape(batch_size, query_len, self.n_heads, self.head_dim)
        k = k.reshape(batch_size, src_len, self.n_heads, self.head_dim)
        v = v.reshape(batch_size, src_len, self.n_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        energy = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            # [batch_size, 1, src_len] -> [batch_size, 1, 1, src_len]
            mask = mask.unsqueeze(1).to(torch.bool)

            fill_value = -1e4 if energy.dtype == torch.float16 else -1e9
            energy = energy.masked_fill(~mask, fill_value)

        attention = F.softmax(energy, dim=-1)
        attention = self.dropout(attention)

        output = torch.matmul(attention, v)

        output = output.transpose(1, 2).contiguous()
        output = output.reshape(batch_size, query_len, self.dec_hidden_dim)

        output = self.norm_out(self.output_proj(output))
        output = output.squeeze(1)

        if need_weights:
            attention = attention[:, :, 0, :]
        else:
            attention = None

        return output, attention


class Decoder(nn.Module):
    def __init__(
        self,
        output_dim,
        embedding_dim,
        enc_output_dim,
        dec_hidden_dim,
        num_layers,
        dropout,
        n_heads,
        pad_idx=0,
    ):
        super().__init__()

        self.output_dim = output_dim
        self.dec_hidden_dim = dec_hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(
            output_dim,
            embedding_dim,
            padding_idx=pad_idx,
        )

        self.embed_norm = nn.LayerNorm(embedding_dim)

        self.attention = MultiHeadAttention(
            enc_output_dim=enc_output_dim,
            dec_hidden_dim=dec_hidden_dim,
            n_heads=n_heads,
            dropout=dropout,
        )

        self.rnn = nn.LSTM(
            embedding_dim + dec_hidden_dim,
            dec_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.rnn_norm = nn.LayerNorm(dec_hidden_dim)

        self.fc_out = nn.Linear(
            dec_hidden_dim + embedding_dim + dec_hidden_dim,
            output_dim,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_token,
        hidden,
        cell,
        encoder_outputs,
        src_mask=None,
        need_attention=False,
    ):
        embedded = self.embedding(input_token)
        embedded = self.embed_norm(embedded)
        embedded = self.dropout(embedded)

        query = hidden[-1]

        context, attn_weights = self.attention(
            query=query,
            key_value=encoder_outputs,
            mask=src_mask,
            need_weights=need_attention,
        )

        context = context.unsqueeze(1)

        rnn_input = torch.cat((embedded, context), dim=-1)

        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        output = self.rnn_norm(output)

        prediction_input = torch.cat((output, embedded, context), dim=-1)

        # Return raw logits. Do not apply LayerNorm or softmax here.
        prediction = self.fc_out(prediction_input)
        prediction = prediction.squeeze(1)

        return prediction, hidden, cell, attn_weights


class Seq2SeqAttentionModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=256,
        hidden_dim=384,
        num_layers=1,
        dropout=0.1,
        n_heads=4,
        pad_idx=0,
    ):
        super().__init__()

        if hidden_dim % 2 != 0:
            raise ValueError("hidden_dim must be even.")

        if hidden_dim % n_heads != 0:
            raise ValueError("hidden_dim must be divisible by n_heads.")

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.pad_idx = pad_idx

        enc_hidden_dim = hidden_dim // 2
        enc_output_dim = enc_hidden_dim * 2

        self.encoder = Encoder(
            input_dim=vocab_size,
            embedding_dim=embedding_dim,
            enc_hidden_dim=enc_hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            pad_idx=pad_idx,
        )

        self.decoder = Decoder(
            output_dim=vocab_size,
            embedding_dim=embedding_dim,
            enc_output_dim=enc_output_dim,
            dec_hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            n_heads=n_heads,
            pad_idx=pad_idx,
        )

        # Usually Identity because enc_output_dim == hidden_dim.
        if enc_output_dim == hidden_dim:
            self.hidden_proj = nn.Identity()
            self.cell_proj = nn.Identity()
        else:
            self.hidden_proj = nn.Linear(enc_output_dim, hidden_dim)
            self.cell_proj = nn.Linear(enc_output_dim, hidden_dim)

        self.hidden_norm = nn.LayerNorm(hidden_dim)
        self.cell_norm = nn.LayerNorm(hidden_dim)

    def make_src_mask(self, src):
        return src.ne(self.pad_idx).unsqueeze(1)

    def forward(
        self,
        noisy_ids,
        clean_ids,
        teacher_forcing_ratio=0.5,
        return_attention=False,
    ):
        """
        noisy_ids:
            [batch_size, src_len]
        clean_ids:
            [batch_size, trg_len]

        returns:
            outputs:
                [batch_size, trg_len, vocab_size]
            attention_weights:
                None during normal training
                or [batch_size, trg_len, n_heads, src_len]
        """
        batch_size = noisy_ids.size(0)
        trg_len = clean_ids.size(1)
        src_len = noisy_ids.size(1)

        device = noisy_ids.device

        src_mask = self.make_src_mask(noisy_ids)

        encoder_outputs, hidden, cell = self.encoder(noisy_ids)

        hidden = torch.tanh(self.hidden_norm(self.hidden_proj(hidden)))
        cell = torch.tanh(self.cell_norm(self.cell_proj(cell)))

        outputs = torch.zeros(
            batch_size,
            trg_len,
            self.vocab_size,
            device=device,
            dtype=encoder_outputs.dtype,
        )

        if return_attention:
            attention_weights = torch.zeros(
                batch_size,
                trg_len,
                self.decoder.attention.n_heads,
                src_len,
                device=device,
                dtype=encoder_outputs.dtype,
            )
        else:
            attention_weights = None

        input_token = clean_ids[:, 0].unsqueeze(1)

        for t in range(1, trg_len):
            output, hidden, cell, attn_weights = self.decoder(
                input_token=input_token,
                hidden=hidden,
                cell=cell,
                encoder_outputs=encoder_outputs,
                src_mask=src_mask,
                need_attention=return_attention,
            )

            outputs[:, t, :] = output

            if return_attention:
                attention_weights[:, t, :, :] = attn_weights

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(dim=1)

            input_token = (
                clean_ids[:, t].unsqueeze(1)
                if teacher_force
                else top1.unsqueeze(1)
            )

        return outputs, attention_weights


def init_weights(model):
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.dim() < 2:
            continue

        if "weight_ih" in name:
            nn.init.xavier_uniform_(param)
        elif "weight_hh" in name:
            nn.init.orthogonal_(param)
        elif "embedding" in name:
            nn.init.xavier_uniform_(param)
        elif "weight" in name:
            nn.init.xavier_uniform_(param)


# ----------------------------
# Train / eval
# ----------------------------

def train_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    epoch,
    num_epochs,
    teacher_forcing_ratio,
    use_amp,
):
    model.train()

    total_loss = 0.0

    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch}/{num_epochs}",
        ncols=120,
        unit="batch",
    )

    for batch_idx, batch in enumerate(progress_bar):
        clean_ids = batch["clean_ids"].to(device)
        noisy_ids = batch["noisy_ids"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device, use_amp):
            outputs, _ = model(
                noisy_ids=noisy_ids,
                clean_ids=clean_ids,
                teacher_forcing_ratio=teacher_forcing_ratio,
                return_attention=False,
            )

            logits = outputs[:, 1:].reshape(-1, outputs.size(-1))
            targets = clean_ids[:, 1:].reshape(-1)

            loss = criterion(logits, targets)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)

        progress_bar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "avg": f"{avg_loss:.4f}",
            }
        )

    return total_loss / max(1, len(dataloader))


@torch.inference_mode()
def evaluate(model, dataloader, criterion, device, use_amp):
    model.eval()

    total_loss = 0.0

    progress_bar = tqdm(
        dataloader,
        desc="Validation",
        ncols=100,
        unit="batch",
    )

    for batch in progress_bar:
        clean_ids = batch["clean_ids"].to(device)
        noisy_ids = batch["noisy_ids"].to(device)

        with autocast_context(device, use_amp):
            outputs, _ = model(
                noisy_ids=noisy_ids,
                clean_ids=clean_ids,
                teacher_forcing_ratio=0.0,
                return_attention=False,
            )

            logits = outputs[:, 1:].reshape(-1, outputs.size(-1))
            targets = clean_ids[:, 1:].reshape(-1)

            loss = criterion(logits, targets)

        total_loss += loss.item()

    return total_loss / max(1, len(dataloader))


# ----------------------------
# Checkpoint helpers
# ----------------------------

def _move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def load_training_checkpoint(
    checkpoint_path: Path,
    model,
    optimizer,
    scheduler,
    device,
):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(model_state_dict)

    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        _move_optimizer_state_to_device(optimizer, device)
    else:
        print("Warning: checkpoint has no optimizer_state_dict; optimizer starts fresh.")

    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    else:
        print("Warning: checkpoint has no scheduler_state_dict; scheduler starts fresh.")

    checkpoint_epoch = int(checkpoint.get("epoch", 0))
    best_val_loss = checkpoint.get("best_val_loss", checkpoint.get("val_loss", float("inf")))

    return checkpoint_epoch, float(best_val_loss)


def save_training_checkpoint(
    checkpoint_path: Path,
    epoch: int,
    model,
    optimizer,
    scheduler,
    val_loss: float,
    best_val_loss: float,
    vocab_size: int,
    pad_idx: int,
    args,
):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "vocab_size": vocab_size,
            "pad_idx": pad_idx,
            "config": vars(args),
        },
        checkpoint_path,
    )


# ----------------------------
# Main
# ----------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-path",
        type=str,
        default=str(PROJECT_ROOT / "data" / "pair" / "train"),
    )

    parser.add_argument(
        "--test-path",
        type=str,
        default=str(PROJECT_ROOT / "data" / "pair" / "test"),
    )

    parser.add_argument(
        "--vocab-path",
        type=str,
        default=str(PROJECT_ROOT / "data" / "pair" / "vocab.json"),
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints"),
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to a checkpoint to continue training from.",
    )

    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)

    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--n-heads", type=int, default=4)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--teacher-forcing", type=float, default=0.5)

    parser.add_argument("--train-workers", type=int, default=8)
    parser.add_argument("--val-workers", type=int, default=4)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--amp", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    train_path = Path(args.train_path)
    test_path = Path(args.test_path)
    vocab_path = Path(args.vocab_path)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MPS-Optimized Attention Seq2Seq Spellchecker")
    print("=" * 80)

    device = get_best_device(force_cpu=args.force_cpu)
    print(f"Device: {device}")

    if device.type == "mps":
        print("Using Apple Silicon MPS backend")

    if args.amp:
        print("AMP: enabled")
    else:
        print("AMP: disabled")

    print("\nLoading tokenizer...")
    tokenizer = Tokenizer(max_length=args.max_length)
    tokenizer.load_vocab(str(vocab_path))

    vocab_size = len(tokenizer.vocab)
    pad_idx = tokenizer.token_to_id("<PAD>")

    print(f"Vocabulary size: {vocab_size}")
    print(f"PAD id: {pad_idx}")

    print("\nLoading data...")
    train_items, train_skipped = load_pairs_from_path(train_path)
    test_items, test_skipped = load_pairs_from_path(test_path)

    if not train_items:
        raise FileNotFoundError(f"No training data found in {train_path}")

    if not test_items:
        raise FileNotFoundError(f"No test data found in {test_path}")

    print(f"Train items: {len(train_items):,} skipped {train_skipped:,}")
    print(f"Test items:  {len(test_items):,} skipped {test_skipped:,}")

    train_dataset = SpellCheckerDataset(
        train_items,
        tokenizer,
        max_length=args.max_length,
    )

    val_dataset = SpellCheckerDataset(
        test_items,
        tokenizer,
        max_length=args.max_length,
    )

    train_loader = make_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.train_workers,
    )

    val_loader = make_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.val_workers,
    )

    print(f"Train batches: {len(train_loader):,}")
    print(f"Val batches:   {len(val_loader):,}")

    print("\nCreating model...")

    model = Seq2SeqAttentionModel(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        n_heads=args.n_heads,
        pad_idx=pad_idx,
    )

    init_weights(model)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    best_val_loss = float("inf")
    start_epoch = 1

    if args.resume_from:
        resume_path = Path(args.resume_from).expanduser()

        print(f"\nLoading checkpoint: {resume_path}")
        checkpoint_epoch, best_val_loss = load_training_checkpoint(
            checkpoint_path=resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        start_epoch = checkpoint_epoch + 1

        print(f"Resumed from epoch: {checkpoint_epoch}")
        print(f"Best validation loss so far: {best_val_loss:.4f}")

    if start_epoch > args.epochs:
        print("\nNothing to train.")
        print(
            f"Checkpoint starts next at epoch {start_epoch}, "
            f"but --epochs is {args.epochs}."
        )
        print("Increase --epochs to continue training beyond the checkpoint.")
        return

    print("\nStarting training...")

    for epoch in range(start_epoch, args.epochs + 1):
        print("\n" + "=" * 80)
        print(f"Epoch {epoch}/{args.epochs}")
        print("=" * 80)

        train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            num_epochs=args.epochs,
            teacher_forcing_ratio=args.teacher_forcing,
            use_amp=args.amp,
        )

        val_loss = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=args.amp,
        )

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        print("\nEpoch summary:")
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Val loss:   {val_loss:.4f}")
        print(f"  LR:         {current_lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            checkpoint_path = checkpoint_dir / "best_attention_model.pt"

            save_training_checkpoint(
                checkpoint_path=checkpoint_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                val_loss=val_loss,
                best_val_loss=best_val_loss,
                vocab_size=vocab_size,
                pad_idx=pad_idx,
                args=args,
            )

            print(f"Saved best model: {checkpoint_path}")
            print(f"Best val loss: {best_val_loss:.4f}")

        latest_checkpoint_path = checkpoint_dir / "latest_attention_checkpoint.pt"
        save_training_checkpoint(
            checkpoint_path=latest_checkpoint_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            val_loss=val_loss,
            best_val_loss=best_val_loss,
            vocab_size=vocab_size,
            pad_idx=pad_idx,
            args=args,
        )
        print(f"Saved latest checkpoint: {latest_checkpoint_path}")

        if epoch % 5 == 0:
            checkpoint_path = checkpoint_dir / f"attention_checkpoint_epoch_{epoch}.pt"

            save_training_checkpoint(
                checkpoint_path=checkpoint_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                val_loss=val_loss,
                best_val_loss=best_val_loss,
                vocab_size=vocab_size,
                pad_idx=pad_idx,
                args=args,
            )

            print(f"Saved checkpoint: {checkpoint_path}")

    print("\n" + "=" * 80)
    print("Training complete")
    print("=" * 80)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved in: {checkpoint_dir}")


if __name__ == "__main__":
    main()
