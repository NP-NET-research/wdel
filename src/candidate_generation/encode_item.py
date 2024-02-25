#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =============================================================================
# @Author  :  xuzf
# @Contact :  xuzhengfei-email@qq.com
# @Create  :  2023-05-26 23:04:46
# @Update  :  2023-10-10 12:25:46
# @Desc    :  None
# =============================================================================

import os
import jsonlines
import pickle
import torch
import wandb
from tqdm import tqdm
from time import strftime, localtime
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

import sys

root_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(root_dir)

from src.candidate_generation.model import Retriever
import src.candidate_generation.prediction as prediction
from src.candidate_generation.tokenization import get_candidate_representation, process_mention_data, entity_text_modeling
from utils.params import Parser
from utils.logger import get_logger
from utils.io import read_dataset, dump_json, load_pickle, dump_pickle


# 读取实体集，包括 title text
def load_entity_dict(logger, path, qid2idx={}, idx2qid=[], debug=False):
    entity_list = []
    logger.info("Loading entity from path: " + path)
    with jsonlines.open(path, "r") as f:
        for line in f:
            qid, title, text = entity_text_modeling(line)
            qid2idx[qid] = len(idx2qid)
            idx2qid.append(qid)
            entity_list.append((title, text))
            if debug and len(entity_list) >= 200:
                break
    logger.info("Loaded {} entities.".format(len(entity_list)))
    # qid2idx_path = path.replace(".jsonl", "_qid2idx.pkl")
    # idx2qid_path = path.replace(".jsonl", "_idx2qid.pkl")
    # with open(qid2idx_path, 'wb') as f:
    #     pickle.dump(qid2idx, f)
    # with open(idx2qid_path, 'wb') as f:
    #     pickle.dump(idx2qid, f)
    # logger.info("Saved qid2idx and idx2qid to {} and {}".format(qid2idx_path, idx2qid_path))
    return entity_list, qid2idx, idx2qid


# 生成 candidate_pool (entity desc -> TensorDataset(input_ids, input_ids))
def get_candidate_pool_tensor(
    entity_desc_list,
    tokenizer,
    max_seq_length,
    logger,
):
    # TODO: add multiple thread process
    logger.info("Convert candidate text to id")
    cand_pool = []
    for entity_desc in tqdm(entity_desc_list, dynamic_ncols=True):
        if type(entity_desc) is tuple:
            title, entity_text = entity_desc
        else:
            title = None
            entity_text = entity_desc

        cand = get_candidate_representation(
            entity_text,
            tokenizer,
            max_seq_length,
            title,
        )
        cand_pool.append(cand["ids"])

    cand_pool = torch.LongTensor(cand_pool)
    return cand_pool


# 编码 candidate_pool
def encode_candidate(retriever, candidate_pool, encode_batch_size, silent):
    retriever.biencoder.eval()
    device = retriever.device
    sampler = SequentialSampler(candidate_pool)
    data_loader = DataLoader(candidate_pool, sampler=sampler, batch_size=encode_batch_size, num_workers=10)
    if silent:
        iter_ = data_loader
    else:
        iter_ = tqdm(data_loader, dynamic_ncols=True)

    cand_encode_list = None
    with torch.no_grad():
        for batch in iter_:
            cands = batch[0].to(device)

            cand_encode = retriever.encode_entity(cands)
            if cand_encode_list is None:
                cand_encode_list = cand_encode
            else:
                cand_encode_list = torch.cat((cand_encode_list, cand_encode))

    return cand_encode_list


# 加载或生成 candidate_pool: entity_num * len(input_ids)
def load_or_generate_candidate_pool(
    tokenizer,
    params,
    logger,
    cand_pool_path,
):
    candidate_pool = None
    if cand_pool_path is not None:
        qid2idx_path = cand_pool_path.replace(".t7", "_qid2idx.pkl")
        idx2qid_path = cand_pool_path.replace(".t7", "_idx2qid.pkl")
        # try to load candidate pool from file
        try:
            logger.info("Loading pre-generated candidate pool from: ")
            logger.info(cand_pool_path)
            candidate_pool = torch.load(cand_pool_path)
            with open(qid2idx_path, "rb") as f:
                qid2idx = pickle.load(f)
            with open(idx2qid_path, "rb") as f:
                idx2qid = pickle.load(f)
        except:
            logger.info("Loading failed. Generating candidate pool")

    if candidate_pool is None:
        # compute candidate pool from entity list
        path = params.get("entity_dict_path", None)
        assert path is not None, "Error! entity_dict_path is empty."
        # Traverse all files in a folder
        if os.path.isfile(path):
            entity_desc_list, qid2idx, idx2qid = load_entity_dict(logger, params["entity_dict_path"])
            candidate_pool = get_candidate_pool_tensor(entity_desc_list, tokenizer, params["max_cand_length"], logger)
        else:
            entity_fpath_list = []
            for file in os.listdir(path):
                if file.endswith(".jsonl"):
                    entity_fpath_list.append(os.path.join(path, file))
            entity_fpath_list = sorted(entity_fpath_list)
            # process each file
            qid2idx = {}
            idx2qid = []
            for file in entity_fpath_list:
                entity_desc_list, qid2idx, idx2qid = load_entity_dict(logger, file, qid2idx, idx2qid, debug=params["debug"])
                if candidate_pool is None:
                    candidate_pool = get_candidate_pool_tensor(entity_desc_list, tokenizer, params["max_cand_length"], logger)
                else:
                    candidate_pool = torch.cat(
                        (
                            candidate_pool,
                            get_candidate_pool_tensor(entity_desc_list, tokenizer, params["max_cand_length"], logger),
                        )
                    )

        candidate_pool = TensorDataset(candidate_pool)
        if cand_pool_path is not None:
            logger.info("Saving candidate pool.")
            torch.save(candidate_pool, cand_pool_path)
            logger.info("Saved candidate pool to " + cand_pool_path)

            with open(qid2idx_path, "wb") as f:
                pickle.dump(qid2idx, f)
            with open(idx2qid_path, "wb") as f:
                pickle.dump(idx2qid, f)
            logger.info("Saved qid2idx and idx2qid to {} and {}".format(qid2idx_path, idx2qid_path))

    return candidate_pool, qid2idx, idx2qid


# 加载并合并 candidate_embedding
def load_and_merge_candidate_embedding(emb_dir, map_dir, save_dir=None):
    split_index_list = sorted([int(file.lstrip("wk_info_").rstrip("_emb.t7")) for file in os.listdir(emb_dir)])
    candidate_emb = None
    qid2idx = None
    idx2qid = None
    for split_index in split_index_list:
        emb_fpath = os.path.join(emb_dir, f"wk_info_{split_index}_emb.t7")
        qid2idx_fpath = os.path.join(map_dir, f"wk_info_{split_index}_pool_qid2idx.pkl")
        idx2qid_fpath = os.path.join(map_dir, f"wk_info_{split_index}_pool_idx2qid.pkl")
        if candidate_emb is None:
            # candidate_emb = torch.load(emb_fpath)
            candidate_emb = 1
            qid2idx = load_pickle(qid2idx_fpath)
            idx2qid = load_pickle(idx2qid_fpath)
        else:
            # new_candidate_emb = torch.load(emb_fpath)
            # new_qid2idx = load_pickle(qid2idx_fpath)
            new_idx2qid = load_pickle(idx2qid_fpath)
            # candidate_emb = torch.cat((candidate_emb, new_candidate_emb))
            for qid in new_idx2qid:
                qid2idx[qid] = len(idx2qid)
                idx2qid.append(qid)
            assert len(qid2idx) == len(idx2qid), f"Error! there is duplicate qid in split {split_index}"
    if save_dir is not None:
        # torch.save(candidate_emb, os.path.join(save_dir, "wk_info_all_emb.t7"))
        dump_pickle(os.path.join(save_dir, "wk_info_all_pool_qid2idx.pkl"), qid2idx)
        dump_pickle(os.path.join(save_dir, "wk_info_all_pool_idx2qid.pkl"), idx2qid)
    emb_fpath = "output/cg/train_retriever_be@01-27-12:40/wk_info_all_emb.t7"
    candidate_emb = torch.load(emb_fpath)
    return candidate_emb, qid2idx, idx2qid


# 对查询样本 tokenization
def tokenize_query(tokenizer, query_samples, max_context_length, max_cand_length, silent=False, logger=None, debug=False):
    _, test_tensor_data = process_mention_data(
        query_samples,
        tokenizer,
        max_context_length=max_context_length,
        max_cand_length=max_cand_length,
        silent=silent,
        just_mention=True,
        logger=logger,
        debug=debug,
    )
    return test_tensor_data


# 对查询样本进行编码
def encode_query(retriever, query_tensor_data, encode_batch_size, silent=True):
    retriever.biencoder.eval()
    device = retriever.device
    sampler = SequentialSampler(query_tensor_data)
    data_loader = DataLoader(query_tensor_data, sampler=sampler, batch_size=encode_batch_size, num_workers=10)
    if silent:
        iter_ = data_loader
    else:
        iter_ = tqdm(data_loader, dynamic_ncols=True)

    query_encode_list = None
    with torch.no_grad():
        for batch in iter_:
            queries = batch[0].to(device)
            query_encode = retriever.encode_mention(queries)
            if query_encode_list is None:
                query_encode_list = query_encode
            else:
                query_encode_list = torch.cat((query_encode_list, query_encode))

    return query_encode_list


# 检索候选集
def retrieve_candidate(retriever, query_encode, candidate_encoding, top_k, silent=False):
    retriever.biencoder.eval()
    device = retriever.device
    query_encode = query_encode.to(device)
    candidate_encoding = candidate_encoding.to(device)
    scores = torch.mm(query_encode, candidate_encoding.t())
    scores = scores.squeeze(0)
    top_k_scores, top_k_indices = torch.topk(scores, top_k, dim=0)
    top_k_scores = top_k_scores.tolist()
    top_k_indices = top_k_indices.tolist()
    return top_k_scores, top_k_indices


def main(params, logger):
    output_path = params["output_path"]
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # save the parameters to file
    dump_json(os.path.join(output_path, "eval_params.json"), params, indent=4, sort_keys=True)

    if params["use_wandb"]:
        wandb.init(project="wdel_eval_cg", name=output_path + "-" + params["mode"], config=params)

    # Init model
    retriever = Retriever(params)
    tokenizer = retriever.tokenizer

    cand_encode_path = params.get("cand_encode_path", None)

    # candidate encoding is not pre-computed.
    # load/generate candidate pool to compute candidate encoding.
    cand_pool_path = params.get("cand_pool_path", None)
    candidate_pool, qid2idx, idx2qid = load_or_generate_candidate_pool(tokenizer, params, logger, cand_pool_path)

    if params["mode"] == "tokenize_candidate":
        return

    # compute candidate encoding.
    candidate_encoding = None
    if cand_encode_path is not None:
        # try to load candidate encoding from path
        # if success, avoid computing candidate encoding
        try:
            logger.info("Loading pre-generated candidate encode path.")
            candidate_encoding = torch.load(cand_encode_path)
        except:
            logger.info("Loading failed. Generating candidate encoding.")

    if candidate_encoding is None:
        candidate_encoding = encode_candidate(retriever, candidate_pool, params["encode_batch_size"], silent=params["silent"])

        if cand_encode_path is not None:
            # Save candidate encoding to avoid re-compute
            logger.info("Saving candidate encoding to file " + cand_encode_path)
            torch.save(candidate_encoding, cand_encode_path)
    if params["mode"] == "encode_candidate":
        logger.info("Candidate embedding is saved to " + cand_encode_path)
        return
    # Evaluate model on test set
    test_samples = read_dataset(params["mode"], params["data_path"], debug=params["debug"])
    logger.info("Read {} test samples from {}".format(len(test_samples), params["mode"]))

    for i in range(len(test_samples)):
        test_samples[i]["label_id"] = qid2idx.get(test_samples[i]["qid"], -1)

    _, test_tensor_data = process_mention_data(
        test_samples,
        tokenizer,
        params["max_context_length"],
        params["max_cand_length"],
        silent=params["silent"],
        logger=logger,
        debug=params["debug"],
    )
    test_sampler = SequentialSampler(test_tensor_data)
    test_dataloader = DataLoader(test_tensor_data, sampler=test_sampler, batch_size=params["eval_batch_size"])

    save_results = params.get("save_topk_result")
    new_data, result = prediction.get_topk_predictions(
        retriever,
        test_dataloader,
        candidate_pool,
        candidate_encoding,
        params["silent"],
        logger,
        params["top_k"],
        save_results,
    )

    if params["use_wandb"]:
        for i in range(result.LEN):
            if result.top_k < result.rank[i]:
                break
            wandb.log({"topk": result.rank[i], "recall@k": result.hits[i] / float(result.cnt)})

    if save_results:
        save_data_dir = os.path.join(output_path, "top%d_candidates" % params["top_k"])
        if not os.path.exists(save_data_dir):
            os.makedirs(save_data_dir)
        save_data_path = os.path.join(save_data_dir, "%s.t7" % params["mode"])
        torch.save(new_data, save_data_path)


if __name__ == "__main__":
    parser = Parser(add_model_args=True)
    parser.add_eval_args()

    args = parser.parse_args()
    print(args)

    params = args.__dict__
    output_path = os.path.join(params["output_path"], "eval_retriever_be@%s" % (strftime("%m-%d-%H:%M", localtime())))
    params["output_path"] = output_path
    logger = get_logger("eval_retriever_be", output_dir=output_path)
    mode_list = params["mode"].split(",")
    for mode in mode_list:
        new_params = params
        new_params["mode"] = mode
        main(new_params, logger)
