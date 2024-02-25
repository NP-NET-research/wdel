import os
import json
import jsonlines
from tqdm import tqdm, trange


import sys

root_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(root_dir)

from src.candidate_generation.tokenization import entity_text_modeling

from utils.io import write_to_file


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


def load_cg_result_from_file(file):
    with open(file, "r") as f:
        data = json.load(f)
    return data


def load_test_dataset(file):
    with jsonlines.open(file, "r") as f:
        data = [line for line in f]
    return data


def process_single_query(query, kb, qid2idx, candidate_qids):
    mention = query["mention"]
    context_left = query["context_left"]
    context_right = query["context_right"]

    gold_entity = query["entity"]
    gold_entity_text = entity_text_modeling(gold_entity)
    gold_entity["text_feature"] = gold_entity_text

    # find candidate qids
    candidates = [kb[qid2idx[qid]] for qid in candidate_qids]
    for idx, candidate in enumerate(candidates):
        _, title, text = entity_text_modeling(candidate)
        candidates[idx]["text_feature"] = "{{" + title + "}}" + "\t" + text

    return mention, context_left, context_right, gold_entity, candidates


def post_process(kb, qid2idx, cg_result, test_data, output_fpath):
    # process test data
    processed_test_data = []
    for query, candidate_qids, label in tqdm(zip(test_data, cg_result["candidates"], cg_result["labels"]), dynamic_ncols=True):
        mention, context_left, context_right, gold_entity, candidates = process_single_query(query, kb, qid2idx, candidate_qids)
        processed_test_data.append(
            {
                "mention": mention,
                "context_left": context_left,
                "context_right": context_right,
                "gold_entity": gold_entity,
                "candidates": candidates,
                "label": label,
            }
        )

    # save processed test data
    # output_fpath = os.path.join(output_dir, "processed_test_data.json")
    with jsonlines.open(output_fpath, "w") as f:
        for item in processed_test_data:
            f.write(item)

    return processed_test_data


def format_output(processed_test_data, output_fpath):
    # format output
    formatted_output = []
    for query in tqdm(processed_test_data, dynamic_ncols=True):
        mention = query["mention"]
        context = query["context_left"] + "[[" + mention + "]]" + query["context_right"]
        gold_entity = query["gold_entity"]["text_feature"]
        gold_entity = "{{" + gold_entity[1] + "}}" + "\t" + gold_entity[2]
        candidates = [str(idx) + ": " + candidate["text_feature"] for idx, candidate in enumerate(query["candidates"])]
        format_item = (
            "Context:"
            + context
            + "\nMention:"
            + mention
            + "\n\tGold Entity:"
            + gold_entity
            + "\n\tGold Label:"
            + str(query["label"])
            + "\n\tCandidates:\n\t\t"
            + "\n\t\t".join(candidates)
            + "\n"
            + "-" * 50 + "\n"
        )
        formatted_output.append(format_item)
    # save formatted output
    # output_fpath = os.path.join(output_dir, "formatted_output.json")
    write_to_file(output_fpath, ''.join(formatted_output))

    return formatted_output


if __name__ == "__main__":
    # example usage

    usage_qids = set()
    
    # load test data
    test_data_fs_fpath = "data/hansel/hansel-test-fs.jsonl"
    test_data_fs = load_test_dataset(test_data_fs_fpath)
    test_data_zs_fpath = "data/hansel/hansel-test-zs.jsonl"
    test_data_zs = load_test_dataset(test_data_zs_fpath)

    # load cg results
    cg_result_fpath = "output/cg/eval_retriever_be@01-25-10:56/top10_candidates/hansel-test-fs.json"
    cg_result_fs = load_cg_result_from_file(cg_result_fpath)
    for qids in cg_result_fs["candidates"]:
        usage_qids.update(qids)

    cg_result_fpath = "output/cg/eval_retriever_be@01-25-10:56/top10_candidates/hansel-test-zs.json"
    cg_result_zs = load_cg_result_from_file(cg_result_fpath)
    for qids in cg_result_zs["candidates"]:
        usage_qids.update(qids)

    cg_result_fpath = "output/cg/eval_retriever_be@01-25-15:06/top10_candidates/hansel-test-fs.json"
    cg_result_fs_hansel = load_cg_result_from_file(cg_result_fpath)
    for qids in cg_result_fs_hansel["candidates"]:
        usage_qids.update(qids)

    cg_result_fpath = "output/cg/eval_retriever_be@01-25-15:06/top10_candidates/hansel-test-zs.json"
    cg_result_zs_hansel = load_cg_result_from_file(cg_result_fpath)
    for qids in cg_result_zs_hansel["candidates"]:
        usage_qids.update(qids)

    # load kb
    kb_dir = "KB/wikidata-2023-12-22/tmp/wk_brief_info"
    kb, qid2idx = load_kb_from_folder(kb_dir, usage_qids)
   
    output_dir = "output/cg/eval_retriever_be@01-25-10:56/top10_candidates"
    # post process
    processed_test_data = post_process(
        kb, qid2idx, cg_result_fs, test_data_fs, output_fpath=os.path.join(output_dir, "processed_fs_data.jsonl")
    )
    # format output
    formatted_output = format_output(processed_test_data, output_fpath=os.path.join(output_dir, "formatted_fs_output.txt"))

    # post process
    processed_test_data = post_process(
        kb, qid2idx, cg_result_zs, test_data_zs, output_fpath=os.path.join(output_dir, "processed_zs_data.jsonl")
    )
    # format output
    formatted_output = format_output(processed_test_data, output_fpath=os.path.join(output_dir, "formatted_zs_output.txt"))

    output_dir = "output/cg/eval_retriever_be@01-25-15:06/top10_candidates"
    # post process
    processed_test_data = post_process(
        kb, qid2idx, cg_result_fs_hansel, test_data_fs, output_fpath=os.path.join(output_dir, "processed_fs_data.jsonl")
    )
    # format output
    formatted_output = format_output(processed_test_data, output_fpath=os.path.join(output_dir, "formatted_fs_output.txt"))

    # post process
    processed_test_data = post_process(
        kb, qid2idx, cg_result_zs_hansel, test_data_zs, output_fpath=os.path.join(output_dir, "processed_zs_data.jsonl")
    )
    # format output
    formatted_output = format_output(processed_test_data, output_fpath=os.path.join(output_dir, "formatted_zs_output.txt"))
