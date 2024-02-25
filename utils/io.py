#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =============================================================================
# @File    :  /el_wikipedia_cn/utils/io.py
# @Author  :  xuzf
# @Contact :  xuzhengfei-email@qq.com
# @Create  :  2022-03-01 15:50:58
# @Update  :  2023-10-10 13:45:55
# @Desc    :  None
# =============================================================================
import os
import json
import jsonlines
import torch
import pickle
from transformers import WEIGHTS_NAME, CONFIG_NAME


# 记录召回率 Recall@k
class Stats:
    def __init__(self, top_k=1000):
        self.cnt = 0
        self.hits = []
        self.top_k = top_k
        self.rank = [1, 4, 8, 10, 16, 32, 64, 100, 128, 256, 512]
        self.LEN = len(self.rank)
        for i in range(self.LEN):
            self.hits.append(0)

    def add(self, idx):
        self.cnt += 1
        if idx == -1:
            return
        for i in range(self.LEN):
            if idx < self.rank[i]:
                self.hits[i] += 1

    def extend(self, stats):
        self.cnt += stats.cnt
        for i in range(self.LEN):
            self.hits[i] += stats.hits[i]

    def output(self):
        output_json = "Total: %d examples." % self.cnt
        for i in range(self.LEN):
            if self.top_k < self.rank[i]:
                break
            output_json += " r@%d: %.4f" % (self.rank[i], self.hits[i] / float(self.cnt))

        return output_json


def read_dataset(dataset_name, preprocessed_json_data_parent_folder, train_size_num=50000, debug=False):
    file_name = "{}.jsonl".format(dataset_name)
    txt_file_path = os.path.join(preprocessed_json_data_parent_folder, file_name)
    is_train = "train" in dataset_name
    samples = []
    with jsonlines.open(txt_file_path, "r") as file:
        for line in file:
            if is_train and len(samples) >= train_size_num:
                break
            samples.append(line)
    return samples

def load_pickle(path):
    with open(path, "rb") as reader:
        return pickle.load(reader)
    
def dump_pickle(path, data):
    with open(path, "wb") as writer:
        pickle.dump(data, writer)

def write_to_file(path, string, mode="w"):
    with open(path, mode) as writer:
        writer.write(string)


def dump_json(path, data, indent=False, sort_keys=False):
    with open(path, "w") as writer:
        writer.write(json.dumps(data, indent=indent, ensure_ascii=False, sort_keys=sort_keys))


def save_model(model, tokenizer, output_dir):
    """Saves the model and the tokenizer used in the output directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_to_save = model
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    # output_config_file = os.path.join(output_dir, CONFIG_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    # model_to_save.config.to_json_file(output_config_file)
    # tokenizer.save_vocabulary(output_dir)

def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()