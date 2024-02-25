import os
import json
import jsonlines
from tqdm import tqdm

# 加载候选生成结果
cg_res_fpath = "output/cg/eval_retriever_be@02-06-12:56/top10_candidates/hansel-dev.json"
with open(cg_res_fpath, "r") as f:
    cg_res = json.load(f)
labels = cg_res["labels"]

# 加载训练语料
train_fpath_raw = "data/hansel/hansel-dev.jsonl"
raw_data = []
with jsonlines.open(train_fpath_raw, "r") as f:
    for line in f:
        raw_data.append(line)
print(f"Loaded {len(raw_data)} lines from {train_fpath_raw}")

# 合并候选实体集合与训练语料
candidates = []
neg_num = 9
for i in range(len(cg_res["candidates"])):
    cands = cg_res["candidates"][i]
    label = labels[i]
    # candidates.append(cands[:neg_num + 1])
    # raw_data[i]["label"] = label
    if label >= 0 and label <= neg_num:
        candidates.append(cands[:neg_num + 1])
        raw_data[i]["label"] = label
    else:
        candidates.append(cands[:neg_num] + [raw_data[i]["entity"]["qid"]])
        raw_data[i]["label"] = neg_num
usage_candidates = []
for cands in candidates:
    usage_candidates.extend(cands)
usage_candidates = set(usage_candidates)
print(f"Used {len(usage_candidates)} candidates")


# 2. 加载实体集
def load_kb_from_folder(dir, usage_qids=None):
    kb = []
    qid2idx = {}
    split_index_list = sorted([int(file.lstrip("wk_info_").rstrip(".jsonl")) for file in os.listdir(dir)])
    for split_index in tqdm(split_index_list, dynamic_ncols=True):
        fpath = os.path.join(dir, f"wk_info_{split_index}.jsonl")
        with jsonlines.open(fpath, "r") as f:
            for entity in f:
                if usage_qids is not None and entity["qid"] not in usage_qids:
                    continue
                qid2idx[entity["qid"]] = len(kb)
                kb.append(entity)
    return kb, qid2idx


kb_dir = "KB/wikidata-2023-12-22/tmp/wk_brief_info"
kb, qid2idx = load_kb_from_folder(kb_dir, usage_qids=usage_candidates)

# 4. 准备训练数据
train_data = []
print(f"Loaded {len(raw_data)} training data")
for i, sample in tqdm(enumerate(raw_data), total=len(raw_data), dynamic_ncols=True):
    cands = []
    flag = True
    for cand in candidates[i]:
        idx = qid2idx.get(cand, -1)
        if idx >= 0:
            cands.append(kb[idx])
        else:
            flag = False
            break
    if flag:
        sample["candidates"] = cands
        train_data.append(sample)
print(f"Prepared {len(train_data)} training data")

# 5. 保存训练数据
train_fpath_target = "data/hansel/hansel-train-dev-h10.jsonl"
with jsonlines.open(train_fpath_target, "w") as f:
    for line in train_data:
        f.write(line)
print(f"Saved {len(train_data)} lines to {train_fpath_target}")
