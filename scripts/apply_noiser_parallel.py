#!/usr/bin/env python3
import sys,os,json,random
from multiprocessing import Pool, cpu_count, Manager
import queue as _pyqueue
from functools import partial
from typing import Any
import importlib.util
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# ensure spellchecker package dir on path
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE not in sys.path:
    sys.path.insert(0, BASE)
# prefer the pruned structured noiser (self-contained) for reliable loading
# fallback to the simpler `smart_noiser.py` implementation available in the repo
NOISER_FILE = os.path.join(BASE, 'noiser', 'smart_noiser.py')

# loader for AzerbaijaniTextNoiser
def load_noiser_class():
    # ensure noiser folder is importable for intra-module imports
    noiser_dir = os.path.dirname(NOISER_FILE)
    if noiser_dir not in sys.path:
        sys.path.insert(0, noiser_dir)
    if BASE not in sys.path:
        sys.path.insert(0, BASE)
    # Pre-load sibling module smart_noiser_structured if present so imports inside
    # the twostage file (e.g. `from smart_noiser_structured import ...`) succeed.
    sibling = os.path.join(noiser_dir, 'smart_noiser_structured.py')
    if os.path.exists(sibling):
        name = 'smart_noiser_structured'
        if name not in sys.modules:
            spec_dep = importlib.util.spec_from_file_location(name, sibling)
            mod_dep = importlib.util.module_from_spec(spec_dep)
            spec_dep.loader.exec_module(mod_dep)
            sys.modules[name] = mod_dep

    spec = importlib.util.spec_from_file_location('noiser_twostage', NOISER_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, 'AzerbaijaniTextNoiser')

# extract candidate text from an item
def extract_text(item: Any):
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ('text','content','body'):
            if key in item and isinstance(item[key], str):
                return item[key]
        # fallback: try first string value
        for v in item.values():
            if isinstance(v, str):
                return v
    return None

# process a single file path
def process_file(path, out_dir, min_times=1, max_times=5, seed=None, progress_q=None):
    random.seed(seed)
    AzerbaijaniTextNoiser = load_noiser_class()
    noiser = AzerbaijaniTextNoiser()
    rel = os.path.basename(path)
    out_path = os.path.join(out_dir, rel + '.noised.jsonl')
    count = 0
    with open(path, 'r', encoding='utf-8') as f, open(out_path, 'w', encoding='utf-8') as out:
        try:
            data = json.load(f)
            items = data if isinstance(data, list) else [data]
        except Exception:
            # fallback to line-wise json / text
            f.seek(0)
            items = []
            for line in f:
                line=line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except Exception:
                    items.append(line)

        for item in items:
            text = extract_text(item)
            if text is None:
                continue
            n = random.randint(min_times, max_times)
            variants = []
            for _ in range(n):
                try:
                    noisy, stats = noiser(text)
                except Exception:
                    try:
                        # some implementations may return just a string
                        noisy = noiser(text)
                    except Exception:
                        noisy = text
                variants.append(noisy)
            out_obj = {'original': text, 'noisy': variants}
            out.write(json.dumps(out_obj, ensure_ascii=False) + '\n')
            # notify progress (one unit per original item processed)
            try:
                if progress_q is not None:
                    progress_q.put(1)
            except Exception:
                pass
            count += 1
    return {'file': path, 'processed': count, 'out': out_path}


def main():
    data_clean = os.path.join(BASE, 'data', 'clean')
    out_dir = os.path.join(BASE, 'data', 'nosied')
    os.makedirs(out_dir, exist_ok=True)
    files = []
    for root,dirs,files_ in os.walk(data_clean):
        for fn in files_:
            if fn.lower().endswith(('.json','.jsonl','.ndjson')):
                files.append(os.path.join(root,fn))
    if not files:
        print('No files found in', data_clean)
        return
    print('Found', len(files), 'files; using up to', cpu_count(), 'processes')
    # pre-scan total items so we can show an overall progress bar
    def count_items_in_file(path):
        c = 0
        try:
            with open(path,'r',encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if extract_text(item) is not None:
                                c += 1
                    else:
                        if extract_text(data) is not None:
                            c = 1
                except Exception:
                    f.seek(0)
                    for line in f:
                        line=line.strip()
                        if not line: continue
                        try:
                            obj=json.loads(line)
                            if extract_text(obj) is not None:
                                c += 1
                        except Exception:
                            c += 1
        except Exception:
            pass
        return c

    total_items = 0
    per_file_counts = {}
    for p in files:
        n = count_items_in_file(p)
        per_file_counts[p] = n
        total_items += n

    procs = min(len(files), cpu_count())
    results = []
    manager = Manager()
    progress_q = manager.Queue()
    if tqdm:
        pbar = tqdm(total=total_items, unit='items')
    else:
        pbar = None
    async_results = []
    with Pool(processes=procs) as pool:
        for p in files:
            ar = pool.apply_async(process_file, (p, out_dir, 1, 5, None, progress_q))
            async_results.append(ar)

        # consume progress updates until all tasks complete
        completed = 0
        total_tasks = len(async_results)
        # continue until all async results are ready
        while True:
            # drain queue
            try:
                while True:
                    progress_q.get_nowait()
                    completed += 1
                    if pbar:
                        pbar.update(1)
            except _pyqueue.Empty:
                pass

            # check if all tasks finished
            if all(ar.ready() for ar in async_results):
                # drain remaining
                try:
                    while True:
                        progress_q.get_nowait()
                        completed += 1
                        if pbar:
                            pbar.update(1)
                except _pyqueue.Empty:
                    pass
                break

    if pbar:
        pbar.close()

    # collect results
    for ar in async_results:
        try:
            res = ar.get()
            results.append(res)
        except Exception as e:
            results.append({'file': None, 'processed': 0, 'out': None, 'error': str(e)})
    # write summary
    summary_path = os.path.join(out_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print('Done. Summary:', summary_path)

if __name__ == '__main__':
    main()