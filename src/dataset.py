<<<<<<< HEAD
"""Dataset classes and data loading utilities."""

import json
import random
import string
from pathlib import Path
from typing import List, Dict, Any, Union, Callable
from functools import partial

import torch
from torch.utils.data import Dataset, DataLoader

from src.tokenizer import Tokenizer

def load_json_data(path: Path, shuffle: bool=True) -> List[Dict[str, str]]:
    """Load and optionally shuffle JSON data."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if shuffle:
            random.shuffle(data)
    return data

def load_json_data_multiple(paths: List[Path], shuffle: bool=True) -> List[Dict[str, str]]:
    """Load and optionally shuffle multiple JSON data files."""
    all_data = []
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_data.extend(data)
                print(f"Loaded {len(data)} examples from {path.name}")
        except Exception as e:
            print(f"Error loading file {path}: {e}")

    if shuffle:
        random.shuffle(all_data)

    print(f"Combined dataset contains {len(all_data)} examples")
    return all_data

class SpellingDataset(Dataset):
    def __init__(
        self,
        paths: Union[Path, List[Path]],
        tokenizer: Tokenizer,
        build_vocab: bool=False,
        vocab_save_path: str='vocab.json',
        truncate: Union[float, None] = None
    ):
        """
        Dataset for spelling correction task.

        Args:
            paths: Path or list of paths to the JSON data files
            tokenizer: Tokenizer instance
            build_vocab: Whether to build vocabulary from this dataset
            vocab_save_path: Path to save vocabulary
            truncate: If not None, use only this fraction of the data
        """
        # Handle both single path and list of paths
        if isinstance(paths, list):
            self.data = load_json_data_multiple(paths, shuffle=True)
        else:
            self.data = load_json_data(paths, shuffle=True)

        self.tokenizer = tokenizer

        if build_vocab:
            # Extract original and noisy texts for vocabulary building
            all_texts = []
            for item in self.data:
                all_texts.append(item["original"].lower())
                all_texts.append(item["noisy"].lower())

            # Build vocabulary with all texts
            self.tokenizer.build_vocab(all_texts)
            self.tokenizer.save_vocab(vocab_save_path)

        if truncate:
            self.data = self.data[:int(len(self.data) * truncate)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch, tokenizer):
    """Collate function for DataLoader."""
    original_texts = [item["original"].lower() for item in batch]
    noisy_texts = [item["noisy"].lower() for item in batch]

    # Tokenize once so the pre-padding lengths can be used for packed LSTM inputs.
    original_encoded = [tokenizer.encode(text) for text in original_texts]
    noisy_encoded = [tokenizer.encode(text) for text in noisy_texts]
    original_lengths = [len(ids) for ids in original_encoded]
    noisy_lengths = [len(ids) for ids in noisy_encoded]

    # Pad all texts to the tokenizer max length for batching.
    original_ids = [tokenizer.pad_sequence(ids) for ids in original_encoded]
    noisy_ids = [tokenizer.pad_sequence(ids) for ids in noisy_encoded]

    # Convert to tensors
    original_ids_tensor = torch.tensor(original_ids, dtype=torch.long)
    noisy_ids_tensor = torch.tensor(noisy_ids, dtype=torch.long)
    original_lengths_tensor = torch.tensor(original_lengths, dtype=torch.long)
    noisy_lengths_tensor = torch.tensor(noisy_lengths, dtype=torch.long)

    return {
        "clean_ids": original_ids_tensor,
        "noisy_ids": noisy_ids_tensor,
        "clean_lengths": original_lengths_tensor,
        "noisy_lengths": noisy_lengths_tensor,
        "clean_texts": original_texts,
        "noisy_texts": noisy_texts
    }

def get_dataloaders(
    train_paths: Union[str, Path, List[Union[str, Path]]],
    val_paths: Union[str, Path, List[Union[str, Path]]],
    tokenizer: Tokenizer,
    batch_size: int = 32,
    build_vocab: bool = True,
    vocab_save_path: str='vocab.json',
    truncate_train: Union[float, None] = None,
    truncate_test: Union[float, None] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
) -> tuple:
    """Create train and validation dataloaders."""

    # Convert to list if single path
    if not isinstance(train_paths, list):
        train_paths = [train_paths]

    if not isinstance(val_paths, list):
        val_paths = [val_paths]

    # Convert string paths to Path objects
    train_paths = [Path(path) if isinstance(path, str) else path for path in train_paths]
    val_paths = [Path(path) if isinstance(path, str) else path for path in val_paths]

    print(f"Training on {len(train_paths)} files: {[p.name for p in train_paths]}")
    print(f"Validating on {len(val_paths)} files: {[p.name for p in val_paths]}")

    train_dataset = SpellingDataset(
        train_paths,
        tokenizer,
        build_vocab=build_vocab,
        truncate=truncate_train,
        vocab_save_path=vocab_save_path
    )

    val_dataset = SpellingDataset(
        val_paths,
        tokenizer,
        truncate=truncate_test
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    return train_dataloader, val_dataloader
=======
"""Dataset classes and data loading utilities."""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Union

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence  # NEW: for dynamic batch padding

from src.tokenizer import Tokenizer


def load_json_data(path: Path, shuffle: bool = True) -> List[Dict[str, str]]:
    """Load and optionally shuffle JSON data."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if shuffle:
            random.shuffle(data)
    return data


def load_json_data_multiple(paths: List[Path], shuffle: bool = True) -> List[Dict[str, str]]:
    """Load and optionally shuffle multiple JSON data files."""
    all_data = []
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_data.extend(data)
                print(f"Loaded {len(data)} examples from {path.name}")
        except Exception as e:
            print(f"Error loading file {path}: {e}")

    if shuffle:
        random.shuffle(all_data)

    print(f"Combined dataset contains {len(all_data)} examples")
    return all_data


class SpellingDataset(Dataset):
    def __init__(
        self,
        paths: Union[Path, List[Path]],
        tokenizer: Tokenizer,
        build_vocab: bool = False,
        vocab_save_path: str = "vocab.json",
        truncate: Union[float, None] = None,
    ):
        """
        Dataset for spelling correction task.

        Args:
            paths: Path or list of paths to the JSON data files
            tokenizer: Tokenizer instance
            build_vocab: Whether to build vocabulary from this dataset
            vocab_save_path: Path to save vocabulary
            truncate: If not None, use only this fraction of the data
        """
        # Handle both single path and list of paths
        if isinstance(paths, list):
            self.data = load_json_data_multiple(paths, shuffle=True)
        else:
            self.data = load_json_data(paths, shuffle=True)

        self.tokenizer = tokenizer

        if build_vocab:
            # Extract clean and noisy texts for vocabulary building
            all_texts = []
            for item in self.data:
                all_texts.append(item["clean"].lower())
                all_texts.append(item["noisy"].lower())

            # Build vocabulary with all texts
            self.tokenizer.build_vocab(all_texts)
            self.tokenizer.save_vocab(vocab_save_path)

        if truncate:
            self.data = self.data[: int(len(self.data) * truncate)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch, tokenizer: Tokenizer):
    """
    Collate function for DataLoader with dynamic padding and length tracking.

    - Encodes each text to variable length ids
    - Pads to the max length *in this batch* (not a global max)
    - Returns sequence lengths so the encoder can use packed sequences
    """
    clean_texts = [item["clean"].lower() for item in batch]
    noisy_texts = [item["noisy"].lower() for item in batch]

    # Encode all texts to variable-length sequences (no padding here)
    clean_id_seqs = [torch.tensor(tokenizer.encode(text), dtype=torch.long)
                     for text in clean_texts]
    noisy_id_seqs = [torch.tensor(tokenizer.encode(text), dtype=torch.long)
                     for text in noisy_texts]

    # Sequence lengths before padding
    clean_lengths = torch.tensor([seq.size(0) for seq in clean_id_seqs], dtype=torch.long)
    noisy_lengths = torch.tensor([seq.size(0) for seq in noisy_id_seqs], dtype=torch.long)

    # Get pad token id from tokenizer
    pad_token_id = tokenizer.token_to_id(tokenizer.special_tokens["pad_token"])

    # Pad to max length within the batch
    # Result shapes: [batch_size, max_seq_len_in_batch]
    clean_ids_tensor = pad_sequence(clean_id_seqs, batch_first=True, padding_value=pad_token_id)
    noisy_ids_tensor = pad_sequence(noisy_id_seqs, batch_first=True, padding_value=pad_token_id)

    return {
        "clean_ids": clean_ids_tensor,
        "clean_lengths": clean_lengths,  # NEW: for pack_padded_sequence
        "noisy_ids": noisy_ids_tensor,
        "noisy_lengths": noisy_lengths,  # NEW: for pack_padded_sequence
        "clean_texts": clean_texts,
        "noisy_texts": noisy_texts,
    }


def get_dataloaders(
    train_paths: Union[str, Path, List[Union[str, Path]]],
    val_paths: Union[str, Path, List[Union[str, Path]]],
    tokenizer: Tokenizer,
    batch_size: int = 32,
    build_vocab: bool = True,
    vocab_save_path: str = "vocab.json",
    truncate_train: Union[float, None] = None,
    truncate_test: Union[float, None] = None,
    num_workers: int = 4,         # NEW: CPU workers for faster loading
    pin_memory: bool = True,      # NEW: page-locked memory for faster H2D copies
) -> tuple:
    """Create train and validation dataloaders."""

    # Convert to list if single path
    if not isinstance(train_paths, list):
        train_paths = [train_paths]

    if not isinstance(val_paths, list):
        val_paths = [val_paths]

    # Convert string paths to Path objects
    train_paths = [Path(path) if isinstance(path, str) else path for path in train_paths]
    val_paths = [Path(path) if isinstance(path, str) else path for path in val_paths]

    print(f"Training on {len(train_paths)} files: {[p.name for p in train_paths]}")
    print(f"Validating on {len(val_paths)} files: {[p.name for p in val_paths]}")

    train_dataset = SpellingDataset(
        train_paths,
        tokenizer,
        build_vocab=build_vocab,
        truncate=truncate_train,
        vocab_save_path=vocab_save_path,
    )

    val_dataset = SpellingDataset(
        val_paths,
        tokenizer,
        truncate=truncate_test,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_dataloader, val_dataloader
>>>>>>> origin/master
