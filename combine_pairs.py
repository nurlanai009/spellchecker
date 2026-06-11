import json
from pathlib import Path
import os

# Load all splits
print("Loading train.json...")
with open('/Users/nurlanmalikov7294/Documents/naic/spellchecker/data/processed/train.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)
print(f"  Train: {len(train_data)} pairs")

print("Loading val.json...")
with open('/Users/nurlanmalikov7294/Documents/naic/spellchecker/data/processed/val.json', 'r', encoding='utf-8') as f:
    val_data = json.load(f)
print(f"  Val: {len(val_data)} pairs")

print("Loading test.json...")
with open('/Users/nurlanmalikov7294/Documents/naic/spellchecker/data/processed/test.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)
print(f"  Test: {len(test_data)} pairs")

# Combine all
all_data = train_data + val_data + test_data
print(f"\nTotal combined: {len(all_data)} pairs")

# Save to single file
output_path = '/Users/nurlanmalikov7294/Documents/naic/spellchecker/data/processed/all_pairs.json'
print(f"\nSaving to {output_path}...")
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(all_data, f, ensure_ascii=False, indent=2)

file_size = os.path.getsize(output_path) / 1024 / 1024
print(f"\n✅ Saved {len(all_data)} pairs to all_pairs.json")
print(f"   File size: {file_size:.2f} MB")
print(f"   Location: {output_path}")
