"""
Collect all data from /pair folder into one JSON file.
Handles both .jsonl and .json formats.
"""

import json
from pathlib import Path

def load_jsonl_data(file_path):
    """Load data from JSONL format."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                # Normalize to clean/noisy format
                if 'original' in item and 'noisy' in item:
                    data.append({
                        'clean': item['original'],
                        'noisy': item['noisy']
                    })
                elif 'clean' in item and 'noisy' in item:
                    data.append(item)
            except:
                continue
    return data

def load_json_data(file_path):
    """Load data from JSON format."""
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

def main():
    pair_dir = Path('/Users/nurlanmalikov7294/Documents/naic/spellchecker/data/pair')
    all_data = []
    
    print("="*80)
    print("Collecting all data from /pair folder")
    print("="*80)
    
    # Process all files
    for file_path in sorted(pair_dir.glob('*')):
        if file_path.is_file():
            print(f"\nProcessing: {file_path.name}")
            
            try:
                if file_path.suffix == '.jsonl' or file_path.name.endswith('.jsonl'):
                    data = load_jsonl_data(file_path)
                elif file_path.suffix == '.json':
                    data = load_json_data(file_path)
                else:
                    print(f"  Skipping unknown format: {file_path.name}")
                    continue
                
                print(f"  Loaded: {len(data)} pairs")
                all_data.extend(data)
                
            except Exception as e:
                print(f"  Error loading {file_path.name}: {e}")
                continue
    
    print(f"\n{'='*80}")
    print(f"Total collected: {len(all_data)} pairs")
    print(f"{'='*80}")
    
    # Save to single file
    output_path = pair_dir / 'all_pairs_combined.json'
    print(f"\nSaving to: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    
    import os
    file_size = os.path.getsize(output_path) / 1024 / 1024
    
    print(f"\n{'='*80}")
    print(f"✅ SUCCESS!")
    print(f"{'='*80}")
    print(f"File: {output_path}")
    print(f"Pairs: {len(all_data):,}")
    print(f"Size: {file_size:.2f} MB")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
