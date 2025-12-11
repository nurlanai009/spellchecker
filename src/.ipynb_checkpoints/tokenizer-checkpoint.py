"""Tokenizer for character-level processing of text."""

import json
from typing import List, Dict, Any
from collections import defaultdict
from tqdm import tqdm
import string

class Tokenizer:
    def __init__(self, max_length: int=256):
        self.max_length = max_length
        self.special_tokens = {
            "pad_token": "<PAD>",
            "unk_token": "<UNK>",
            "bos_token": "<BOS>",
            "eos_token": "<EOS>",
        }
        self.vocab = None
        self.reverse_vocab = None
        self.vocab_stats = None
        # Azerbaijani lowercase alphabet
        self.az_alphabet = "abcçdeəfgğhxıijkqlmnoöprsştuüvyz"

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into a list of characters with special tokens."""
        return [self.special_tokens["bos_token"]] + list(text) + [self.special_tokens["eos_token"]]

    def __call__(self, text: str) -> List[str]:
        """Process text and handle truncation if needed."""
        tokens = self.tokenize(text)

        # Handle truncation if tokens exceed max_length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length-1] + [self.special_tokens["eos_token"]]

        return tokens

    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs."""
        tokens = self.__call__(text)
        ids = [self.token_to_id(token) for token in tokens]
        return ids

    def decode(self, ids: List[int]) -> str:
        """Convert token IDs back to text."""
        tokens = [self.id_to_token(id) for id in ids]
        # Filter out special tokens except for unknown tokens
        text = "".join([t for t in tokens if t not in [
            self.special_tokens["pad_token"],
            self.special_tokens["bos_token"],
            self.special_tokens["eos_token"]
        ]])
        return text

    def build_vocab(self, texts: List[str]) -> Dict[str, int]:
        """
        Build vocabulary from list of texts, ensuring it includes Azerbaijani lowercase letters.
        """
        vocab = {}
        self.vocab_stats = defaultdict(lambda: 0)
        
        # Add special tokens first
        for token in self.special_tokens.values():
            if token not in vocab:
                vocab[token] = len(vocab)
        
        # Add all Azerbaijani lowercase characters
        for char in self.az_alphabet:
            if char not in vocab:
                vocab[char] = len(vocab)
        
        # Add digits and punctuation if needed
        for char in string.digits + string.punctuation:
            if char not in vocab:
                vocab[char] = len(vocab)
                
        # Count character frequencies in texts
        for text in tqdm(texts, desc="Analyzing Vocabulary"):
            for char in text:
                if char in vocab:
                    self.vocab_stats[char] += 1
                
        self.vocab = vocab
        self.reverse_vocab = {v: k for k, v in vocab.items()}
        print("\nVocabulary has been built", f"\nLength of the vocabulary: {len(self.vocab)}")
        return vocab

    def token_to_id(self, token: str) -> int:
        """Convert token to ID."""
        if not hasattr(self, 'vocab') or self.vocab is None:
            raise ValueError("Vocabulary not built. Call build_vocab first.")
        return self.vocab.get(token, self.vocab[self.special_tokens["unk_token"]])

    def id_to_token(self, id: int) -> str:
        """Convert ID to token."""
        if not hasattr(self, 'reverse_vocab') or self.reverse_vocab is None:
            raise ValueError("Vocabulary not built. Call build_vocab first.")
        return self.reverse_vocab.get(id, self.special_tokens["unk_token"])

    def pad_sequence(self, token_ids: List[int]) -> List[int]:
        """Pad sequence to max_length."""
        if len(token_ids) >= self.max_length:
            return token_ids[:self.max_length]

        pad_id = self.token_to_id(self.special_tokens["pad_token"])
        padding = [pad_id] * (self.max_length - len(token_ids))
        return token_ids + padding
    
    def save_vocab(self, path: str) -> None:
        """Save vocabulary to a file."""
        if self.vocab is None:
            raise ValueError("Vocabulary not built. Call build_vocab first.")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, indent=4, ensure_ascii=False)
            
    def load_vocab(self, path: str) -> None:
        """Load vocabulary from a file."""
        with open(path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        self.reverse_vocab = {int(v): k for k, v in self.vocab.items()}
        print(f"Loaded vocabulary with {len(self.vocab)} tokens")