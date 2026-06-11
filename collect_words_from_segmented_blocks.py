import json
import random
import re


data = json.load(open(r"/Users/nurlanmalikov7294/Documents/naic/spellchecker/segmented_test.json", 'r', encoding='utf-8'))

def words_from_json(data):
    words = []

    for item in data:
        for block in item.get("blocks", []):
            text = block.get("input_text", "")

            # Unicode-friendly word extraction
            block_words = re.findall(r"[^\W\d_]+", text, flags=re.UNICODE)

            words.extend(block_words)

    return words

words = words_from_json(data)
train_ratio = 0.75
test_ratio = 0.25

train = random.sample(population=words, k=int(len(words)*train_ratio))
test = random.sample(population=words, k=int(len(words)*test_ratio))

with open("200k_words_train.json", "w" , encoding='utf-8') as df:
    json.dump(train, df, ensure_ascii=False, indent=4)


with open("200k_words_test.json", "w" , encoding='utf-8') as df:
    json.dump(test, df, ensure_ascii=False, indent=4)