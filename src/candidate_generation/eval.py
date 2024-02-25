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
from src.candidate_generation.encode_item import (
    load_or_generate_candidate_pool,
    load_and_merge_candidate_embedding,
    encode_candidate,
)
from src.candidate_generation.tokenization import get_candidate_representation, process_mention_data, entity_text_modeling
from utils.params import Parser
from utils.logger import get_logger
from utils.io import read_dataset, dump_json, load_pickle


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

    # candidate encoding is not pre-computed.
    # load/generate candidate pool to compute candidate encoding.
    cand_pool_path = params.get("cand_pool_path", None)
    if cand_pool_path is not None and cand_pool_path.endswith(".t7"):
        candidate_pool, qid2idx, idx2qid = load_or_generate_candidate_pool(tokenizer, params, logger, cand_pool_path)
    else:
        candidate_pool, qid2idx, idx2qid = None, None, None

    if params["mode"] == "tokenize_candidate":
        return

    # compute candidate encoding.
    cand_encode_path = params.get("cand_encode_path", None)
    candidate_encoding = None
    if cand_encode_path is not None:
        # try to load candidate encoding from path
        # if success, avoid computing candidate encoding
        try:
            logger.info("Loading pre-generated candidate encode path.")
            if os.path.isdir(cand_encode_path) and os.path.isdir(cand_pool_path):
                candidate_encoding, qid2idx, idx2qid = load_and_merge_candidate_embedding(
                    cand_encode_path, cand_pool_path, save_dir=os.path.dirname(cand_encode_path)
                )
            else:
                candidate_encoding = torch.load(cand_encode_path)
                qid2idx = load_pickle(cand_encode_path.replace("_emb.t7", "_pool_qid2idx.pkl"))
                idx2qid = load_pickle(cand_encode_path.replace("_emb.t7", "_pool_idx2qid.pkl"))
        except:
            logger.info("Loading failed. Generating candidate encoding.")

    if candidate_encoding is None:
        assert candidate_pool is not None, "Error! Candidate pool is not provided."
        candidate_encoding = encode_candidate(retriever, candidate_pool, params["encode_batch_size"], silent=params["silent"])

        if cand_encode_path is not None:
            # Save candidate encoding to avoid re-compute
            logger.info("Saving candidate encoding to file " + cand_encode_path)
            torch.save(candidate_encoding, cand_encode_path)
    if params["mode"] == "encode_candidate":
        logger.info("Candidate embedding is saved to " + cand_encode_path)
        return

    # Evaluate model on test set
    test_samples = read_dataset(params["mode"], params["data_path"], debug=params["debug"], train_size_num=1000000)
    logger.info("Read {} test samples from {}".format(len(test_samples), params["mode"]))

    for i in range(len(test_samples)):
        test_samples[i]["label_id"] = qid2idx.get(test_samples[i]["uri"], -1)

    _, test_tensor_data = process_mention_data(
        test_samples,
        tokenizer,
        params["max_context_length"],
        params["max_cand_length"],
        silent=params["silent"],
        logger=logger,
        debug=params["debug"],
    )
    # torch.save(test_tensor_data, "output/cg/eval_retriever_be@02-04-12:19/dev_tensor_data.pt")
    # test_tensor_data = torch.load("output/cg/eval_retriever_be@02-04-12:19/train_1M_tensor_data.pt")
    test_sampler = SequentialSampler(test_tensor_data)
    test_dataloader = DataLoader(test_tensor_data, sampler=test_sampler, batch_size=params["eval_batch_size"])

    save_results = params.get("save_topk_result")
    new_data, result = prediction.get_topk_predictions(
        retriever,
        test_dataloader,
        idx2qid,
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
        save_data_path = os.path.join(save_data_dir, "%s.json" % params["mode"])
        dump_json(save_data_path, new_data, indent=4)


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
