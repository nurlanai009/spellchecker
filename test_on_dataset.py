import json
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent / "src"))
from tokenizer import Tokenizer


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
        files = sorted([p for p in path.iterdir() if p.is_file() and p.suffix in {".json", ".jsonl"}])
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
    def __init__(self, data_items):
        self.data = data_items

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Seq2SeqModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=384, num_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.encoder = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.decoder = nn.LSTM(
            embedding_dim,
            hidden_dim * 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc_out = nn.Linear(hidden_dim * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, noisy_ids, max_len, bos_id):
        batch_size = noisy_ids.size(0)
        vocab_size = self.fc_out.out_features

        noisy_emb = self.dropout(self.embedding(noisy_ids))
        _, (hidden, cell) = self.encoder(noisy_emb)

        hidden = hidden.view(self.decoder.num_layers, 2, batch_size, -1)
        hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=2)
        cell = cell.view(self.decoder.num_layers, 2, batch_size, -1)
        cell = torch.cat([cell[:, 0], cell[:, 1]], dim=2)

        outputs = torch.zeros(batch_size, max_len, vocab_size, device=noisy_ids.device)
        input_token = torch.tensor([[bos_id]], device=noisy_ids.device).repeat(batch_size, 1)

        for t in range(1, max_len):
            input_emb = self.dropout(self.embedding(input_token))
            output, (hidden, cell) = self.decoder(input_emb, (hidden, cell))
            prediction = self.fc_out(output)
            outputs[:, t] = prediction.squeeze(1)
            top1 = prediction.argmax(2)
            input_token = top1

        return outputs


def predict_batch(model, tokenizer, noisy_texts, device):
    pad_id = tokenizer.token_to_id("<PAD>")
    bos_id = tokenizer.token_to_id("<BOS>")

    batch_ids = []
    for text in noisy_texts:
        ids = tokenizer.encode(text)
        if hasattr(ids, "ids"):
            ids = ids.ids
        ids = ids[:tokenizer.max_length]
        ids = ids + [pad_id] * (tokenizer.max_length - len(ids))
        batch_ids.append(ids)

    noisy_tensor = torch.tensor(batch_ids, dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(noisy_tensor, tokenizer.max_length, bos_id)
        pred_ids = outputs.argmax(-1).tolist()

    return [tokenizer.decode(ids) for ids in pred_ids]


def main():
    test_path = Path("/Users/nurlanmalikov7294/Documents/naic/spellchecker/data/test")
    vocab_path = Path("/Users/nurlanmalikov7294/Documents/naic/spellchecker/data/pair/vocab.json")
    checkpoint_path = Path("/Users/nurlanmalikov7294/Documents/naic/spellchecker/checkpoints/best_model.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = Tokenizer(max_length=64)
    tokenizer.load_vocab(str(vocab_path))

    model = Seq2SeqModel(vocab_size=len(tokenizer.vocab)).to(device)
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_items, test_skipped = load_pairs_from_path(test_path)
    if not test_items:
        raise FileNotFoundError(f"No test data found in {test_path}")

    print(f"Test items: {len(test_items)} (skipped {test_skipped})")

    dataset = SpellCheckerDataset(test_items)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    for batch in loader:
        clean = batch["clean"]
        noisy = batch["noisy"]
        fixed = predict_batch(model, tokenizer, noisy, device)
        for c, n, f in zip(clean, noisy, fixed):
            print(json.dumps({"clean": c, "noisy": n, "fixed": f}, ensure_ascii=False))
        break


if __name__ == "__main__":
    main()
