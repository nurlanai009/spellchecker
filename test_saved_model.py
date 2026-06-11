import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from tokenizer import Tokenizer

# Change this to your training file name, for example:
# from train_mps import Seq2SeqAttentionModel, SpellCheckerDataset, load_pairs_from_path, evaluate
from train_attention_mps import Seq2SeqAttentionModel, SpellCheckerDataset, load_pairs_from_path, evaluate


MAX_LENGTH = 64

vocab_path = Path("/Users/nurlanmalikov7294/Documents/naic/spellchecker/data/pair/vocab.json")
test_path = Path("/Users/nurlanmalikov7294/Documents/naic/spellchecker/data/pair/test")
checkpoint_path = Path("/Users/nurlanmalikov7294/Documents/naic/spellchecker/checkpoints/best_attention_model.pt")


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def token_id(tokenizer, names):
    for name in names:
        try:
            value = tokenizer.token_to_id(name)
            if value is not None:
                return value
        except Exception:
            pass
    return None


def decode_ids(tokenizer, ids):
    pad_id = token_id(tokenizer, ["<PAD>"])
    bos_id = token_id(tokenizer, ["<BOS>", "<SOS>", "<START>"])
    eos_id = token_id(tokenizer, ["<EOS>", "</s>", "<END>"])

    cleaned = []
    for idx in ids:
        if idx == eos_id:
            break
        if idx in {pad_id, bos_id, None}:
            continue
        cleaned.append(idx)

    try:
        return tokenizer.decode(cleaned)
    except Exception:
        return "".join(tokenizer.id_to_token(i) for i in cleaned)

def correct_text(model, tokenizer, text, device, max_length=64, use_amp=False):
    model.eval()

    enc = tokenizer.encode(text)
    ids = enc.ids if hasattr(enc, "ids") else enc

    pad_id = tokenizer.token_to_id("<PAD>")
    bos_id = token_id(tokenizer, ["<BOS>", "<SOS>", "<START>"])
    eos_id = token_id(tokenizer, ["<EOS>", "</s>", "<END>"])

    if bos_id is None:
        # Fallback, but proper BOS is better.
        bos_id = ids[0] if len(ids) > 0 else pad_id

    noisy_ids = ids[:max_length]
    noisy_ids = noisy_ids + [pad_id] * (max_length - len(noisy_ids))
    noisy_ids = torch.tensor([noisy_ids], dtype=torch.long).to(device)

    # Dummy target sequence.
    # Your model.forward() needs clean_ids only to know max_len and start decoding.
    clean_ids = [bos_id] + [pad_id] * (max_length - 1)
    clean_ids = torch.tensor([clean_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        amp_enabled = use_amp and device.type in {"mps", "cuda"}

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled
        ):
            try:
                outputs = model(
                    noisy_ids,
                    clean_ids,
                    teacher_forcing_ratio=0.0
                )
            except TypeError:
                outputs = model(noisy_ids, clean_ids)

        # Some attention models return: outputs, attentions
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]

        pred_ids = outputs.argmax(dim=-1)[0].tolist()

    # Usually position 0 is empty/PAD because decoding starts at t=1
    pred_ids = pred_ids[1:]

    return decode_ids(tokenizer, pred_ids)


def main():
    device = get_device()
    print(f"Using device: {device}")

    tokenizer = Tokenizer(max_length=MAX_LENGTH)
    tokenizer.load_vocab(str(vocab_path))

    vocab_size = len(tokenizer.vocab)

    model = Seq2SeqAttentionModel(
        vocab_size=vocab_size,
        embedding_dim=256,
        hidden_dim=384,
        num_layers=1,
        dropout=0.1,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Best saved val loss: {checkpoint['val_loss']:.4f}")

    # Test on validation/test set
    test_items, skipped = load_pairs_from_path(test_path)
    test_dataset = SpellCheckerDataset(test_items, tokenizer, max_length=MAX_LENGTH)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=4,
    )

    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("<PAD>"))
    # test_loss = evaluate(model, test_loader, criterion, device, use_amp=True)

    # print(f"Test loss: {test_loss:.4f}")

    

    while True:
        noisy = input("Enter: ")
        corrected = correct_text(model, tokenizer, noisy, device, max_length=MAX_LENGTH)
        print(f"Noisy:     {noisy}")
        print(f"Corrected: {corrected}")
        print()


if __name__ == "__main__":
    main()