import jsonlines
from tqdm import tqdm

train_fpath_raw = "data/hansel/raw/hansel-zero-shot-v1.jsonl"
train_fpath_target = "data/hansel/hansel-test-zs.jsonl"
qid_known_fpath = "data/hansel/wikidata/qids-known-label_desc_alt_P31_P279_map.jsonl"
qid_new_fpath = "data/hansel/wikidata/qids-new-label_desc_alt_P31_P279_map.jsonl"

# 1. 加载原始数据
raw_data = []
with jsonlines.open(train_fpath_raw, 'r') as f:
    for line in f:
        raw_data.append(line)
print(f"Loaded {len(raw_data)} lines from {train_fpath_raw}")

# 2. 加载实体集
kb2idx = {}
kb = []
with jsonlines.open(qid_new_fpath, 'r') as f:
    for line in f:
        kb2idx[line['qid']] = len(kb)
        kb.append(line)
with jsonlines.open(qid_known_fpath, 'r') as f:
    for line in f:
        kb2idx[line['qid']] = len(kb)
        kb.append(line)

# 3. 加载重定向文件
import pickle as pickle
sameAs_map = pickle.load(open("KB/wikidata-2023-12-22/tmp/sameAs_map.pkl", "rb"))

# 4. 准备训练数据
train_data = []
for line in tqdm(raw_data):
    qid = line['gold_id']
    qid = sameAs_map.get(qid, qid)

    context_left = line['text'][:line['start']]
    context_right = line['text'][line['end']:]
    mention = line['mention']

    if qid in kb2idx:
        train_data.append({
            'id': line['id'],
            "context_left": context_left,
            "mention": mention,
            "context_right": context_right,
            "uri": qid,
            "label_id": 0,
            "entity": kb[kb2idx[qid]],
        })
print(f"Prepared {len(train_data)} training data")

# 5. 保存训练数据
with jsonlines.open(train_fpath_target, 'w') as f:
    for line in train_data:
        f.write(line)
print(f"Saved {len(train_data)} lines to {train_fpath_target}")
