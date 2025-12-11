"""Dataset classes and data loading utilities."""

import json
import random
import string
from pathlib import Path
from typing import List, Dict, Any, Union, Callable

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

class SpellingDataset(Dataset):
    def __init__(
        self, 
        path: Path, 
        tokenizer: Tokenizer,
        build_vocab: bool=False,
        truncate: Union[float, None] = None
    ):
        """
        Dataset for spelling correction task.
        
        Args:
            path: Path to the JSON data file
            tokenizer: Tokenizer instance
            build_vocab: Whether to build vocabulary from this dataset
            truncate: If not None, use only this fraction of the data
        """
        self.data = load_json_data(path, shuffle=True)
        self.tokenizer = tokenizer
        
        if build_vocab:
            # Build vocabulary using the Azerbaijani alphabet-focused method
            self.tokenizer.build_vocab([d["clean"] for d in self.data])
            self.tokenizer.save_vocab('data/tokenizer_vocab.json')
            
        if truncate:
            self.data = self.data[:int(len(self.data) * truncate)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch, tokenizer):
    """Collate function for DataLoader."""
    clean_texts = [item["clean"].lower() for item in batch]
    noisy_texts = [item["noisy"].lower() for item in batch]

    # Tokenize and pad all texts
    clean_ids = [tokenizer.pad_sequence(tokenizer.encode(text)) for text in clean_texts]
    noisy_ids = [tokenizer.pad_sequence(tokenizer.encode(text)) for text in noisy_texts]

    # Convert to tensors
    clean_ids_tensor = torch.tensor(clean_ids, dtype=torch.long)
    noisy_ids_tensor = torch.tensor(noisy_ids, dtype=torch.long)

    return {
        "clean_ids": clean_ids_tensor,
        "noisy_ids": noisy_ids_tensor,
        "clean_texts": clean_texts,
        "noisy_texts": noisy_texts
    }

def get_dataloaders(
    train_path: Union[str, Path],
    val_path: Union[str, Path],
    tokenizer: Tokenizer,
    batch_size: int = 32,
    build_vocab: bool = True,
    truncate_train: Union[float, None] = None,
    truncate_test: Union[float, None] = None
) -> tuple:
    """Create train and validation dataloaders."""
    
    train_dataset = SpellingDataset(
        Path(train_path),
        tokenizer,
        build_vocab=build_vocab,
        truncate=truncate_train
    )

    val_dataset = SpellingDataset(
        Path(val_path),
        tokenizer,
        truncate=truncate_test
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )
    
    return train_dataloader, val_dataloader