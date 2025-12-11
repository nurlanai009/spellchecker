"""
Script to train a tokenizer on all available data files and save the vocabulary.
This explicitly loads all data files from the specified structure.
"""

import json
import os
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict

from src.tokenizer import Tokenizer
from src.config import TOKENIZER_VOCAB_PATH, MAX_LENGTH

def load_json_data(path: Path) -> List[Dict[str, str]]:
    """Load JSON data from a file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"Loaded {len(data)} examples from {path.name}")
            return data
    except Exception as e:
        print(f"Error loading file {path}: {e}")
        return []

def extract_texts_from_data(data: List[Dict[str, str]]) -> List[str]:
    """Extract all clean and noisy texts from the data."""
    texts = []
    for item in data:
        if "clean" in item:
            texts.append(item["clean"].lower())
        if "noisy" in item:
            texts.append(item["noisy"].lower())
    return texts

def train_tokenizer(
    data_dir: str = "data", 
    vocab_path: str = TOKENIZER_VOCAB_PATH,
    overwrite: bool = False
):
    """
    Train a tokenizer on all available data files and save the vocabulary.
    
    Args:
        data_dir: Directory containing train and test data
        vocab_path: Path to save the vocabulary file
        overwrite: Whether to overwrite an existing vocabulary file
    """
    # Check if vocabulary file already exists
    if os.path.exists(vocab_path) and not overwrite:
        print(f"Vocabulary file {vocab_path} already exists. Use overwrite=True to replace it.")
        return
    
    # Initialize tokenizer
    tokenizer = Tokenizer(max_length=MAX_LENGTH)
    
    # Explicitly list all data files based on the directory structure
    data_files = [
        # Training data
        Path(data_dir) / "train" / "train_sentences_azeri.json",
        Path(data_dir) / "train" / "train_sentences_english.json",
        Path(data_dir) / "train" / "train_sentences_russian.json",
        Path(data_dir) / "train" / "train_words_azeri.json",
        Path(data_dir) / "train" / "train_words_english.json",
        Path(data_dir) / "train" / "train_words_russian.json",
        
        # Testing data
        Path(data_dir) / "test" / "test_sentences_azeri.json",
        Path(data_dir) / "test" / "test_sentences_english.json",
        Path(data_dir) / "test" / "test_sentences_russian.json",
        Path(data_dir) / "test" / "test_words_azeri.json",
        Path(data_dir) / "test" / "test_words_english.json",
        Path(data_dir) / "test" / "test_words_russian.json"
    ]
    
    # Filter out non-existent files
    existing_files = [f for f in data_files if f.exists()]
    
    if not existing_files:
        print(f"No data files found in {data_dir}")
        return
    
    print(f"Found {len(existing_files)} out of {len(data_files)} expected data files")
    
    # Load data from all files
    all_texts = []
    total_examples = 0
    
    print("\nLoading data files:")
    for file_path in existing_files:
        data = load_json_data(file_path)
        texts = extract_texts_from_data(data)
        all_texts.extend(texts)
        total_examples += len(data)
    
    print(f"\nExtracted {len(all_texts)} texts from {total_examples} examples")
    
    # Build vocabulary
    print("\nBuilding vocabulary...")
    tokenizer.build_vocab(all_texts)
    
    # Save vocabulary
    tokenizer.save_vocab(vocab_path)
    print(f"Tokenizer vocabulary with {len(tokenizer.vocab)} tokens saved to {vocab_path}")
    
    # Print some statistics
    if tokenizer.vocab_stats:
        print("\nMost common characters:")
        sorted_chars = sorted(tokenizer.vocab_stats.items(), key=lambda x: x[1], reverse=True)
        for char, count in sorted_chars[:20]:
            if char in ['\n', '\t', '\r', ' ']:
                char_display = repr(char)
            else:
                char_display = f"'{char}'"
            print(f"{char_display}: {count}")

def main():
    """Main function to train tokenizer."""
    # Train tokenizer with all data files
    train_tokenizer(
        data_dir="data",
        vocab_path=TOKENIZER_VOCAB_PATH,
        overwrite=True  # Set to False to avoid overwriting existing vocab
    )
    
    # Load the trained tokenizer to verify
    tokenizer = Tokenizer(max_length=MAX_LENGTH)
    tokenizer.load_vocab(TOKENIZER_VOCAB_PATH)
    
    # Test the tokenizer on examples from each language
    test_examples = {
        "Azerbaijani": [
            "salam necəsən?",
            "bakıda hava nəcədir?",
            "azərbaycan qəşəng ölkədir."
        ],
        "English": [
            "hello, how are you?",
            "what is the weather like in london?",
            "england is a beautiful country."
        ],
        "Russian": [
            "привет, как дела?",
            "какая погода в москве?",
            "россия красивая страна."
        ]
    }
    
    print("\nTesting tokenizer with examples from each language:")
    for language, examples in test_examples.items():
        print(f"\n{language} examples:")
        for text in examples:
            tokens = tokenizer(text)
            ids = tokenizer.encode(text)
            decoded = tokenizer.decode(ids)
            
            print(f"\nOriginal:  {text}")
            print(f"Tokenized: {tokens[:10]}..." if len(tokens) > 10 else f"Tokenized: {tokens}")
            print(f"Encoded:   {ids[:10]}..." if len(ids) > 10 else f"Encoded: {ids}")
            print(f"Decoded:   {decoded}")

if __name__ == "__main__":
    main()