#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =============================================================================
# @Author  :  xuzf
# @Contact :  xuzhengfei-email@qq.com
# @Create  :  2023-05-26 23:06:48
# @Update  :  2023-07-19 00:06:56
# @Desc    :  None
# =============================================================================

import os
import sys
import faiss
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List

root_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(root_dir)

from src.candidate_generation.encode_item import tokenize_query, encode_query

from utils.io import Stats, dump_json


def get_topk_predictions(
    retriever,
    train_dataloader,
    idx2qid,
    cand_encode_list,
    silent,
    logger,
    top_k=10,
    save_predictions=False,
):
    retriever.biencoder.eval()
    device = retriever.device
    logger.info("Getting top %d predictions." % top_k)
    if silent:
        iter_ = train_dataloader
    else:
        iter_ = tqdm(train_dataloader)

    nn_mention = []
    nn_candidates = []
    nn_confidence = []
    nn_labels = []
    nn_cands_idx = []
    nn_id = []

    res = Stats(top_k)
    with torch.no_grad():
        for step, batch in enumerate(iter_):
            batch = tuple(t.to(device) for t in batch)

            # context_input, _, label_ids = batch
            mention_input, _, label_ids = batch

            scores = retriever.score_candidate(mention_input, None, cand_pool_emb=cand_encode_list.to(device))
            scores, indicies = scores.topk(top_k)
            scores = torch.softmax(scores, dim=-1)

            for i in range(mention_input.size(0)):
                inds = indicies[i]
                score = scores[i]
                pointer = -1  # top-k 中第 pointer 个实体正确命中
                for j in range(top_k):
                    if inds[j].item() == label_ids[i].item():
                        pointer = j
                        break
                res.add(pointer)

                # if pointer == -1:
                #     continue

                if not save_predictions:
                    continue

                # add examples in new_data
                inds = inds.cpu().tolist()
                cur_candidates = [idx2qid[ind] for ind in inds]
                # nn_mention.append(mention_input[i].cpu().tolist())
                nn_candidates.append(cur_candidates)
                nn_confidence.append(score.cpu().tolist())
                nn_labels.append(pointer)
                nn_cands_idx.append(inds)
                nn_id.append(label_ids[i].item())
            if save_predictions and (step + 1) % 200000 == 0:
                nn_data = {
                    # 'mention_vecs': nn_mention,
                    "result": res.output(),
                    "candidates": nn_candidates,
                    "labels": nn_labels,
                    "confidence": nn_confidence,
                    "candidate_id": nn_cands_idx,
                    "id": nn_id,
                }
                logger.info(f"{step + 1} step save checkpoint")
                logger.info(res.output())
                dump_json(f"tmp/train_cg_res_{step+1}.json", nn_data, indent=4)
        logger.info(res.output())
        # nn_mention = torch.LongTensor(nn_mention)
        # nn_candidates = torch.LongTensor(nn_candidates)
        # nn_labels = torch.LongTensor(nn_labels)
        # nn_confidence = torch.FloatTensor(nn_confidence)
        # nn_cands_idx = torch.LongTensor(nn_cands_idx)
        # nn_id = torch.LongTensor(nn_id)

    nn_data = {
        # 'mention_vecs': nn_mention,
        "result": res.output(),
        "candidates": nn_candidates,
        "labels": nn_labels,
        "confidence": nn_confidence,
        "candidate_id": nn_cands_idx,
        "id": nn_id,
    }

    return nn_data, res


def build_faiss_flat_index(cand_encode_list, save_fpath):
    d = cand_encode_list.size(1)
    index = faiss.IndexFlatIP(d)
    index.add(cand_encode_list.cpu().numpy())
    faiss.write_index(index, save_fpath)
    return index


def build_faiss_hnsw_index(cand_encode_list, save_fpath):
    d = cand_encode_list.size(1)
    index = faiss.IndexHNSWFlat(d, 32)
    index.add(cand_encode_list.cpu().numpy())
    faiss.write_index(index, save_fpath)
    return index


def load_faiss_index(save_fpath):
    index = faiss.read_index(save_fpath)
    return index


def hnsw_search(index: faiss.IndexHNSWFlat, query_emb: np.ndarray, top_k=10, efSearch=1024):
    index.hnsw.efSearch = efSearch
    _, I = index.search(query_emb, top_k)
    return I


def get_topk_candidates_faiss(
    retriever,
    querys: List[Dict],
    idx2qid: List,
    index: faiss.IndexHNSWFlat,
    silent: bool = True,
    logger=None,
    top_k: int = 10,
):
    """
    querys: List[Dict], example: [{'context_left': 'What is the capital of ', 'mention': 'France', 'context_right': '?'}]
    idx2qid: List, example: ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
    index: faiss.IndexHNSWFlat
    silent: bool = True
    logger=None
    top_k: int = 10
    """
    query_tensor_data = tokenize_query(
        retriever.tokenizer,
        querys,
        max_context_length=128,
        max_cand_length=128,
        silent=silent,
        logger=logger,
    )
    query_emb = encode_query(
        retriever,
        query_tensor_data,
        encode_batch_size=512,
        silent=silent,
    )
    try:
        query_emb = query_emb.cpu().numpy()
        candidates_index_list = hnsw_search(index, query_emb, top_k=top_k)
        candidates_list = []
        for candidates_index in candidates_index_list:
            candidates = [idx2qid[i] for i in candidates_index]
            candidates_list.append(candidates)
        assert len(candidates_list) == len(querys), "candidates_list length not equal to querys length"
    except Exception as e:
        if logger is not None:
            logger.error(f"Error in get_topk_candidates_faiss: {e}")
        else:
            print(f"Error in get_topk_candidates_faiss: {e}")
        candidates_list = [None for _ in range(len(querys))]

    return candidates_list
