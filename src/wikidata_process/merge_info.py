import os
from tqdm import tqdm, trange
import pickle as pkl
import jsonlines

import sys
root_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(root_dir)

from utils.logger import get_logger
logger = get_logger("merge_info", output_dir="output/data_process", stdout=True)


def dump_pkl(obj, fpath):
    with open(fpath, "wb") as f:
        pkl.dump(obj, f)


def dump_jsonl(obj, fpath):
    with jsonlines.open(fpath, "w") as f:
        if isinstance(obj, list):
            for item in obj:
                f.write(item)
        elif isinstance(obj, dict):
            for key, value in obj.items():
                f.write({"qid": key, **value})


# 1. 读取全部的qids
def get_qid_list(qid_fpath, sameAs_map):
    with open(qid_fpath, "r") as f:
        qids = [line.strip() for line in f.readlines()][1:]  # remove the tsv head
    logger.info(f"read {len(qids)} qids from {qid_fpath}")
    normal_qids = set()
    for qid in qids:
        normal_qids.add(sameAs_map.get(qid, qid))
    logger.info(f"keep {len(normal_qids)} unique qids after redirect normalization")
    return qids


# 3. 初始化最后保留的数据结构
def init_wk_info(qids):
    wk_info = {}
    for qid in tqdm(qids):
        wk_info[qid] = {
            "label": {"en": None, "zh": None},
            "desc": {"en": None, "zh": None},
            "alt": {"en": [], "zh": []},
            "P31": {"qid": [], "en": [], "zh": []},
            "P279": {"qid": [], "en": [], "zh": []},
        }
    return wk_info


# 4. 读取label desc alt信息
def merge_label(label_map, wk_info):
    for qid in wk_info.keys():
        label = label_map.get(qid, None)
        if label is None:
            continue
        wk_info[qid]["label"]["en"] = label["label"]["en"]
        wk_info[qid]["label"]["zh"] = label["label"]["zh"]
    return wk_info


def merge_desc(desc_map, wk_info):
    for qid in wk_info.keys():
        desc = desc_map.get(qid, None)
        if desc is None:
            continue
        wk_info[qid]["desc"]["en"] = desc["desc"]["en"]
        wk_info[qid]["desc"]["zh"] = desc["desc"]["zh"]
    return wk_info


def merge_alt(alt_map, wk_info):
    for qid in wk_info.keys():
        alt = alt_map.get(qid, None)
        if alt is None:
            continue
        wk_info[qid]["alt"]["en"] = alt["alt"]["en"]
        wk_info[qid]["alt"]["zh"] = alt["alt"]["zh"]
    return wk_info


# 4.1 删除label和desc为空的条目
def remove_empty_label_desc(wk_info):
    for qid in list(wk_info.keys()):
        if (
            wk_info[qid]["label"]["en"] is None
            and wk_info[qid]["label"]["zh"] is None
            and wk_info[qid]["desc"]["en"] is None
            and wk_info[qid]["desc"]["zh"] is None
        ):
            del wk_info[qid]
    return wk_info


def merge_P31(wk_info, P31_fpath, wikimedia_filter_set):
    with open(P31_fpath, "r") as f:
        P31_list = [line for line in f.readlines()][1:]
        for line in tqdm(P31_list):
            qids = line.strip().split("\t")
            qids = [sameAs_map.get(qid, qid) for qid in qids]
            if qids[0] not in wk_info:
                continue
            if qids[1] in wikimedia_filter_set:
                wk_info.pop(qids[0], None)
                continue
            elif len(wk_info[qids[0]]["P31"]["qid"]) > 10:
                continue
            single_p31_path_qid = []
            single_p31_path_en = []
            single_p31_path_zh = []
            for qid in qids[1:]:
                single_p31_path_qid.append(qid)
                label = label_map.get(qid, None)
                if label is None:
                    continue
                if not label["label"]["en"] is None:
                    single_p31_path_en.append(label["label"]["en"])
                if not label["label"]["zh"] is None:
                    single_p31_path_zh.append(label["label"]["zh"])
            wk_info[qids[0]]["P31"]["qid"].append(single_p31_path_qid)
            wk_info[qids[0]]["P31"]["en"].append(single_p31_path_en)
            wk_info[qids[0]]["P31"]["zh"].append(single_p31_path_zh)
    return wk_info


def merge_P279(wk_info, P279_fpath, wikimedia_filter_set):
    with open(P279_fpath, "r") as f:
        P279_list = [line for line in f.readlines()][1:]
        for line in tqdm(P279_list):
            qids = line.strip().split("\t")
            qids = [sameAs_map.get(qid, qid) for qid in qids]
            if qids[0] not in wk_info:
                continue
            if qids[1] in wikimedia_filter_set:
                wk_info.pop(qids[0], None)
                continue
            elif len(wk_info[qids[0]]["P279"]["qid"]) > 10:
                continue
            single_p279_path_qid = []
            single_p279_path_en = []
            single_p279_path_zh = []
            for qid in qids[1:]:
                single_p279_path_qid.append(qid)
                label = label_map.get(qid, None)
                if label is None:
                    continue
                if not label["label"]["en"] is None:
                    single_p279_path_en.append(label["label"]["en"])
                if not label["label"]["zh"] is None:
                    single_p279_path_zh.append(label["label"]["zh"])
            wk_info[qids[0]]["P279"]["qid"].append(single_p279_path_qid)
            wk_info[qids[0]]["P279"]["en"].append(single_p279_path_en)
            wk_info[qids[0]]["P279"]["zh"].append(single_p279_path_zh)
    return wk_info


def handle_chunk(qids, wikimedia_filter_set):
    wk_info = init_wk_info(qids)
    logger.info(f"processing {len(qids)} qids")
    wk_info = merge_label(label_map, wk_info)
    wk_info = merge_desc(desc_map, wk_info)
    wk_info = remove_empty_label_desc(wk_info)
    logger.info(f"keep {len(wk_info)} qids after removing empty label and desc")
    wk_info = merge_alt(alt_map, wk_info)
    wk_info = merge_P31(wk_info, "KB/wikidata-2023-12-22/tmp/P31-hop3.txt", wikimedia_filter_set)
    logger.info(f"keep {len(wk_info)} qids after merging P31")
    wk_info = merge_P279(wk_info, "KB/wikidata-2023-12-22/tmp/P279-hop3.txt", wikimedia_filter_set)
    logger.info(f"keep {len(wk_info)} qids after merging P279")
    return wk_info


if __name__ == "__main__":

    label_fpath = "KB/wikidata-2023-12-22/tmp/all_label_map.pkl"
    desc_fpath = "KB/wikidata-2023-12-22/tmp/all_desc_map.pkl"
    alt_fpath = "KB/wikidata-2023-12-22/tmp/all_alt_map.pkl"
    redirect_fpath = "KB/wikidata-2023-12-22/tmp/sameAs_map.pkl"
    wikimedia_filter_fpath = "KB/wikimedia_internal_qid.json"

    qid_fpath = "KB/wikidata-2023-12-22/tmp/all_qid.txt"
    qid_split_name = qid_fpath.split("/")[-1].split(".")[0]

    with open(label_fpath, "rb") as f:
        label_map = pkl.load(f)
    with open(desc_fpath, "rb") as f:
        desc_map = pkl.load(f)
    with open(alt_fpath, "rb") as f:
        alt_map = pkl.load(f)
    with open(redirect_fpath, "rb") as f:
        sameAs_map = pkl.load(f)
    import json
    with open(wikimedia_filter_fpath, 'r') as f:
        wikimedia_filter = json.load(f)
        wikimedia_filter_set = set()
        for qids in wikimedia_filter.values():
            wikimedia_filter_set.update(set(qids))

    qids = get_qid_list(qid_fpath, sameAs_map=sameAs_map)

    for i in trange(0, len(qids), 10000000, dynamic_ncols=True):
        qids_chunk = qids[i : i + 10000000]
        wk_info = handle_chunk(qids_chunk, wikimedia_filter_set)
        with jsonlines.open(f"KB/wikidata-2023-12-22/tmp/wk_brief_info/wk_info_{i}.jsonl", "w") as f:
            for key, value in wk_info.items():
                f.write({"qid": key, **value})
        del wk_info


