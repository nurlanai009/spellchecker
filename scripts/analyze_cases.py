import os,sys,json,re
from collections import Counter
root = os.path.abspath(os.path.dirname(__file__))
folders=['clean','noisy','pair']
file_exts=('.json','.jsonl','.ndjson')

# Regexes
months = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?|yanvar|fevral|mart|aprel|may|iyun|iyul|avqust|sentyabr|oktyabr|noyabr|dekabr)"
re_date = re.compile(r"\b(?:\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}[./-]\d{1,2}[./-]\d{1,2}|\d{1,2}\s+" + months + r"(?:,?\s+\d{4})?)\b", re.I)
re_range = re.compile(r"\b\d+\s*[-–—]\s*\d+\b|\b\d+\s+to\s+\d+\b", re.I)
re_hier = re.compile(r"\b\d+(?:\.\d+){1,}\b")
re_tel = re.compile(r"(?=(?:.*\d){7,})(?:\+?[\d\-\.\s()]{7,})")
re_float = re.compile(r"\b\d+[.,]\d+\b")
re_ordinal = re.compile(r"\b\d+(?:st|nd|rd|th|-ci|-cı|-cu|-cü)\b", re.I)
re_int = re.compile(r"\b\d+\b")
punct_chars = r"!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~–—…«»“”‚"
re_punct = re.compile('[' + re.escape(punct_chars) + ']')
re_list_num_paren = re.compile(r"\b\d+\)")
re_list_num_dot = re.compile(r"\b\d+\.\b")
re_list_letter_paren = re.compile(r"\b[a-zA-Z]\)")

summary = Counter()
per_folder = {f:Counter() for f in folders}
strings_scanned = 0


def extract_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from extract_strings(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from extract_strings(item)

for folder in folders:
    dirp = os.path.join(root,folder)
    if not os.path.isdir(dirp):
        continue
    for dirpath,dirnames,filenames in os.walk(dirp):
        for fn in filenames:
            if not fn.lower().endswith(file_exts):
                continue
            path = os.path.join(dirpath,fn)
            try:
                if fn.lower().endswith('.json'):
                    with open(path,'r',encoding='utf-8') as f:
                        data = json.load(f)
                    items = list(extract_strings(data))
                else:
                    items = []
                    with open(path,'r',encoding='utf-8') as f:
                        for line in f:
                            line=line.strip()
                            if not line:
                                continue
                            try:
                                obj=json.loads(line)
                                items.extend(list(extract_strings(obj)))
                            except Exception:
                                items.append(line)
            except Exception:
                items=[]
                try:
                    with open(path,'r',encoding='utf-8',errors='ignore') as f:
                        for line in f:
                            items.append(line.strip())
                except Exception:
                    continue

            for s in items:
                if not isinstance(s,str):
                    continue
                strings_scanned += 1
                floats = re_float.findall(s)
                if floats:
                    summary['float_count'] += len(floats)
                    per_folder[folder]['float_count'] += len(floats)
                dates = re_date.findall(s)
                if dates:
                    summary['date_count'] += len(dates)
                    per_folder[folder]['date_count'] += len(dates)
                ranges = re_range.findall(s)
                if ranges:
                    summary['range_count'] += len(ranges)
                    per_folder[folder]['range_count'] += len(ranges)
                h = re_hier.findall(s)
                if h:
                    summary['hier_count'] += len(h)
                    per_folder[folder]['hier_count'] += len(h)
                t = re_tel.findall(s)
                if t:
                    summary['tel_count'] += len(t)
                    per_folder[folder]['tel_count'] += len(t)
                o = re_ordinal.findall(s)
                if o:
                    summary['ordinal_count'] += len(o)
                    per_folder[folder]['ordinal_count'] += len(o)
                ints = re_int.findall(s)
                if ints:
                    int_count = 0
                    for m in ints:
                        in_float=False
                        for fval in floats:
                            if m in fval:
                                in_float=True
                                break
                        if not in_float:
                            int_count += 1
                    if int_count:
                        summary['int_count'] += int_count
                        per_folder[folder]['int_count'] += int_count
                puncs = re_punct.findall(s)
                if puncs:
                    summary['punct_count'] += len(puncs)
                    per_folder[folder]['punct_count'] += len(puncs)
                ln1 = re_list_num_paren.findall(s)
                ln2 = re_list_num_dot.findall(s)
                ln3 = re_list_letter_paren.findall(s)
                lm = len(ln1)+len(ln2)+len(ln3)
                if lm:
                    summary['list_marker_count'] += lm
                    per_folder[folder]['list_marker_count'] += lm

# Prepare final summary
out_root = root
out = {
    'total_strings_scanned': strings_scanned,
    'counts': dict(summary),
    'per_folder': {k:dict(v) for k,v in per_folder.items()},
}
perc = {}
for k,v in summary.items():
    perc[k+'_pct'] = (v/strings_scanned*100) if strings_scanned>0 else 0.0
out['percentages_of_strings'] = perc

out_path = os.path.join(out_root,'analysis_cases_summary.json')
with open(out_path,'w',encoding='utf-8') as f:
    json.dump(out,f,ensure_ascii=False,indent=2)
print('ANALYSIS_DONE', out_path)
