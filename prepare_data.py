"""
Collect all spellchecker data, create train/val/test splits, and train tokenizer.
Optimized for M4 Pro MPU training.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers
from tokenizers.processors import TemplateProcessing


def load_jsonl_data(file_path: Path) -> List[Dict[str, str]]:
    """Load data from JSONL format (original/noisy)."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                if 'original' in item and 'noisy' in item:
                    data.append({
                        'clean': item['original'],
                        'noisy': item['noisy']
                    })
            except:
                continue
    return data


def load_json_data(file_path: Path) -> List[Dict[str, str]]:
    """Load data from JSON format (clean/noisy)."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Normalize to clean/noisy format
    normalized = []
    for item in data:
        if isinstance(item, dict):
            if 'clean' in item and 'noisy' in item:
                normalized.append(item)
            elif 'original' in item and 'noisy' in item:
                normalized.append({
                    'clean': item['original'],
                    'noisy': item['noisy']
                })
    return normalized


def collect_all_data(data_dir: Path) -> List[Dict[str, str]]:
    """Collect all data from directory."""
    all_data = []
    
    for file_path in data_dir.glob('*.jsonl'):
        print(f"Loading {file_path.name}...")
        data = load_jsonl_data(file_path)
        all_data.extend(data)
        print(f"  Loaded {len(data)} pairs")
    
    for file_path in data_dir.glob('*.json'):
        if 'filtered' in file_path.name:
            continue  # Skip filtered files
        print(f"Loading {file_path.name}...")
        data = load_json_data(file_path)
        all_data.extend(data)
        print(f"  Loaded {len(data)} pairs")
    
    return all_data


def clean_and_filter_data(data: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Clean and filter the data."""
    cleaned = []
    
    for item in data:
        clean = item.get('clean', '').strip()
        noisy = item.get('noisy', '').strip()
        
        # Skip empty or very short
        if len(clean) < 2 or len(noisy) < 2:
            continue
        
        # Skip if clean and noisy are identical
        if clean == noisy:
            continue
        
        # Skip if too long (limit to 512 chars)
        if len(clean) > 512 or len(noisy) > 512:
            continue
        
        cleaned.append({
            'clean': clean,
            'noisy': noisy
        })
    
    return cleaned


def split_data(data: List[Dict[str, str]], 
               train_ratio: float = 0.8,
               val_ratio: float = 0.1,
               test_ratio: float = 0.1,
               seed: int = 42) -> Tuple[List, List, List]:
    """Split data into train/val/test sets."""
    random.seed(seed)
    random.shuffle(data)
    
    total = len(data)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    return train_data, val_data, test_data


def save_splits(train_data, val_data, test_data, output_dir: Path):
    """Save train/val/test splits."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    splits = {
        'train.json': train_data,
        'val.json': val_data,
        'test.json': test_data
    }
    
    for filename, data in splits.items():
        output_path = output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(data)} examples to {output_path}")


def train_tokenizer(train_data: List[Dict[str, str]], 
                    output_path: Path,
                    vocab_size: int = 10000):
    """Train BPE tokenizer on the data."""
    print(f"\nTraining tokenizer with vocab_size={vocab_size}...")
    
    # Collect all text (both clean and noisy)
    all_text = []
    for item in train_data:
        all_text.append(item['clean'])
        all_text.append(item['noisy'])
    
    # Initialize BPE tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
    
    # Set normalizer (lowercase + NFD unicode normalization)
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFD(),
        normalizers.Lowercase(),
        normalizers.StripAccents()
    ])
    
    # Set pre-tokenizer (split on whitespace and punctuation)
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    
    # Set up trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<PAD>", "<BOS>", "<EOS>", "<UNK>"],
        min_frequency=2
    )
    
    # Train
    tokenizer.train_from_iterator(all_text, trainer=trainer)
    
    # Add post-processor
    tokenizer.post_processor = TemplateProcessing(
        single="<BOS> $A <EOS>",
        special_tokens=[
            ("<BOS>", tokenizer.token_to_id("<BOS>")),
            ("<EOS>", tokenizer.token_to_id("<EOS>")),
        ],
    )
    
    # Save
    tokenizer.save(str(output_path))
    print(f"Tokenizer saved to {output_path}")
    
    return tokenizer


def print_statistics(data: List[Dict[str, str]], name: str):
    """Print dataset statistics."""
    print(f"\n{name} Statistics:")
    print(f"  Total examples: {len(data)}")
    
    clean_lengths = [len(item['clean']) for item in data]
    noisy_lengths = [len(item['noisy']) for item in data]
    
    print(f"  Clean text length: min={min(clean_lengths)}, max={max(clean_lengths)}, avg={sum(clean_lengths)/len(clean_lengths):.1f}")
    print(f"  Noisy text length: min={min(noisy_lengths)}, max={max(noisy_lengths)}, avg={sum(noisy_lengths)/len(noisy_lengths):.1f}")


if __name__ == "__main__":
    print("=" * 80)
    print("Spellchecker Data Preparation")
    print("=" * 80)
    
    # Paths
    data_dir = Path("/Users/nurlanmalikov7294/Documents/naic/spellchecker/data/pair")
    output_dir = Path("/Users/nurlanmalikov7294/Documents/naic/spellchecker/data/processed")
    tokenizer_path = output_dir / "tokenizer.json"
    
    # Step 1: Collect all data
    print("\n📥 Step 1: Collecting data...")
    all_data = collect_all_data(data_dir)
    print(f"Total collected: {len(all_data)} pairs")
    
    # Step 2: Clean and filter
    print("\n🧹 Step 2: Cleaning and filtering...")
    cleaned_data = clean_and_filter_data(all_data)
    print(f"After cleaning: {len(cleaned_data)} pairs ({len(all_data) - len(cleaned_data)} removed)")
    
    # Step 3: Split data
    print("\n✂️  Step 3: Splitting data (80/10/10)...")
    train_data, val_data, test_data = split_data(cleaned_data)
    
    print_statistics(train_data, "Training")
    print_statistics(val_data, "Validation")
    print_statistics(test_data, "Test")
    
    # Step 4: Save splits
    print("\n💾 Step 4: Saving splits...")
    save_splits(train_data, val_data, test_data, output_dir)
    
    # Step 5: Train tokenizer
    print("\n🔤 Step 5: Training tokenizer...")
    tokenizer = train_tokenizer(train_data, tokenizer_path, vocab_size=10000)
    
    # Step 6: Test tokenizer
    print("\n🧪 Step 6: Testing tokenizer...")
    test_texts = [
        "Salam, necəsən?",
        "Bu bir test cümləsidir.",
        "Mən 25 yaşındayam."
    ]
    
    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded.ids)
        print(f"  Original: {text}")
        print(f"  Decoded:  {decoded}")
        print(f"  Tokens:   {encoded.tokens}")
    
    print("\n" + "=" * 80)
    print("✅ Data preparation complete!")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")
    print(f"  - train.json ({len(train_data)} examples)")
    print(f"  - val.json ({len(val_data)} examples)")
    print(f"  - test.json ({len(test_data)} examples)")
    print(f"  - tokenizer.json")
    print("\nNext step: Run training with train.py")
