# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import gc
import concurrent.futures
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset


ENT_START_TAG = "[unused1]"
ENT_END_TAG = "[unused2]"
ENT_TITLE_TAG = "[unused3]"


def select_field(data, key1, key2=None):
    if key2 is None:
        return [example[key1] for example in data]
    else:
        return [example[key1][key2] for example in data]


def entity_text_modeling_en_zh(entity_info):
    """
    建模实体的文本特征，输入为字典，输出为(qid, title, desc)
    entity_info 形如：
    ```json
    {
        "qid": "Q3061828",
        "label": { "en": "Micralestes fodori", "zh": "福氏小鲑脂鲤"},
        "desc": {"en": "species of fish", "zh": null},
        "alt": {"en": [], "zh": []},
        "P31": {
            "qid": [
                ["Q16521", "Q24017414", "Q24017465"]
            ],
            "en": [
                ["taxon", "second-order class", "third-order class"]
            ],
            "zh": [
                ["生物分类单元", "二阶类", "三阶类"]
            ]
        },
        "P279": {"qid": [], "en": [], "zh": []}
    }
    ```
    """
    en_info = ""
    # label
    en_label = entity_info["label"]["en"]
    if not en_label is None:
        en_info += "name: " + en_label + ";"
    # description
    en_desc = entity_info["desc"]["en"]
    if not en_desc is None:
        en_info += "description: " + en_desc + ";"
    # alias table
    en_alt = "/".join(entity_info["alt"]["en"])
    if len(en_alt) > 0:
        en_info += "alias: " + en_alt + ";"
    # P31
    for P31_path in entity_info["P31"]["en"]:
        if len(P31_path) > 0:
            en_info += "instance_of: " + "/".join(P31_path) + ";"
    # P279
    for P279_path in entity_info["P279"]["en"]:
        if len(P279_path) > 0:
            en_info += "subclass_of: " + "/".join(P279_path) + ";"

    zh_info = ""
    # label
    zh_label = entity_info["label"]["zh"]
    if not zh_label is None:
        zh_info += "名称：" + zh_label + "；"
    # description
    zh_desc = entity_info["desc"]["zh"]
    if not zh_desc is None:
        zh_info += "描述：" + zh_desc + "；"
    # alias table
    zh_alt = "/".join(entity_info["alt"]["zh"])
    if len(zh_alt) > 0:
        zh_info += "别名：" + zh_alt + "；"
    # P31
    for P31_path in entity_info["P31"]["zh"]:
        if len(P31_path) > 0:
            zh_info += "是以下实体的实例：" + "/".join(P31_path) + "；"
    # P279
    for P279_path in entity_info["P279"]["zh"]:
        if len(P279_path) > 0:
            zh_info += "是以下实体的子类：" + "/".join(P279_path) + "；"

    title = ""
    if not en_label is None:
        title += en_label
    if not zh_label is None:
        title += "/" + zh_label

    desc = en_info + zh_info

    if len(title) + len(desc) == 0:
        title = entity_info["qid"]
        desc = entity_info["qid"]
    return entity_info["qid"], title.lower(), desc.lower()


def entity_text_modeling(entity_info):
    """
    建模实体的文本特征，输入为字典，输出为(qid, title, desc)
    entity_info 形如：
    ```json
    {
        "qid": "Q3061828",
        "label": { "en": "Micralestes fodori", "zh": "福氏小鲑脂鲤"},
        "desc": {"en": "species of fish", "zh": null},
        "alt": {"en": [], "zh": []},
        "P31": {
            "qid": [
                ["Q16521", "Q24017414", "Q24017465"]
            ],
            "en": [
                ["taxon", "second-order class", "third-order class"]
            ],
            "zh": [
                ["生物分类单元", "二阶类", "三阶类"]
            ]
        },
        "P279": {"qid": [], "en": [], "zh": []}
    }
    ```
    """
    desc = ""
    # label
    zh_label = entity_info["label"]["zh"]
    if not zh_label is None:
        desc += "名称：" + zh_label + "；"
    en_label = entity_info["label"]["en"]
    if not en_label is None:
        desc += "name: " + en_label + ";"

    # description
    zh_desc = entity_info["desc"]["zh"]
    if not zh_desc is None:
        desc += "描述：" + zh_desc + "；"
    en_desc = entity_info["desc"]["en"]
    if not en_desc is None:
        desc += "description: " + en_desc + ";"

    # alias table
    zh_alt = "/".join(entity_info["alt"]["zh"])
    if len(zh_alt) > 0:
        desc += "别名：" + zh_alt + "；"
    en_alt = "/".join(entity_info["alt"]["en"])
    if len(en_alt) > 0:
        desc += "alias: " + en_alt + ";"

    # P31
    for P31_path in entity_info["P31"]["zh"]:
        if len(P31_path) > 0:
            desc += "是以下实体的实例：" + "/".join(P31_path) + "；"
    for P31_path in entity_info["P31"]["en"]:
        if len(P31_path) > 0:
            desc += "instance_of: " + "/".join(P31_path) + ";"
    # P279
    for P279_path in entity_info["P279"]["zh"]:
        if len(P279_path) > 0:
            desc += "是以下实体的子类：" + "/".join(P279_path) + "；"
    for P279_path in entity_info["P279"]["en"]:
        if len(P279_path) > 0:
            desc += "subclass_of: " + "/".join(P279_path) + ";"

    title = ""
    if not zh_label is None:
        title += zh_label
    if not en_label is None:
        title += "/" + en_label

    if len(title) + len(desc) == 0:
        title = entity_info["qid"]
        desc = entity_info["qid"]
    return entity_info["qid"], title.lower(), desc.lower()


def get_context_representation(
    sample,
    tokenizer,
    max_seq_length,
    mention_key="mention",
    context_key="context",
    ent_start_token=ENT_START_TAG,
    ent_end_token=ENT_END_TAG,
):
    mention_tokens = []
    if sample[mention_key] and len(sample[mention_key]) > 0:
        max_mention_length = 16  # (max_seq_length - 1) // 2 - 2
        mention_tokens = tokenizer.tokenize(sample[mention_key])[:max_mention_length]
    # if len(mention_tokens) + 6 >= (max_seq_length):
    #     mention_input_ids = tokenizer.convert_tokens_to_ids(["[CLS]"] + mention_tokens + ["[SEP]"])
    #     mention_input_ids = mention_input_ids + [0] * (max_seq_length - len(mention_input_ids))
    #     return {"tokens": ["[CLS]"] + mention_tokens + ["[SEP]"], "ids": mention_input_ids}

    mention_tokens = [ent_start_token] + mention_tokens + [ent_end_token]
    context_left = sample[context_key + "_left"]
    context_right = sample[context_key + "_right"]
    context_left = tokenizer.tokenize(context_left)
    context_right = tokenizer.tokenize(context_right)

    # max_context_length = max_seq_length - len(mention_tokens)
    max_context_length = max_seq_length

    left_quota = (max_context_length - len(mention_tokens)) // 2 - 1
    right_quota = max_context_length - len(mention_tokens) - left_quota - 2

    left_add = len(context_left)
    right_add = len(context_right)
    if left_add <= left_quota:
        if right_add > right_quota:
            right_quota += left_quota - left_add
    else:
        if right_add <= right_quota:
            left_quota += right_quota - right_add

    context_tokens = context_left[-left_quota:] + mention_tokens + context_right[:right_quota]
    # mention_context_tokens = ["[CLS]"] + mention_tokens[1:-1] + ["[SEP]"] + context_tokens + ["[SEP]"]
    mention_context_tokens = ["[CLS]"] + context_tokens + ["[SEP]"]

    input_ids = tokenizer.convert_tokens_to_ids(mention_context_tokens)
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    try:
        assert len(input_ids) == max_seq_length
    except:
        print(len(mention_tokens))
        print(" ".join(mention_tokens))
        print(len(mention_context_tokens))
        print(" ".join(mention_context_tokens))
        print(len(input_ids))
        print(max_context_length)
        print(max_seq_length)
        exit()

    return {"tokens": mention_context_tokens, "ids": input_ids}


def get_candidate_representation(
    candidate_desc,
    tokenizer,
    max_seq_length,
    candidate_title,
    title_tag=ENT_TITLE_TAG,
):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token

    title_tokens = [cls_token] + tokenizer.tokenize(candidate_title)[:16] + [sep_token]

    max_desc_length = max_seq_length - len(title_tokens)
    cand_tokens = tokenizer.tokenize(candidate_desc)[: max_desc_length - 1] + [sep_token]

    title_cand_tokens = title_tokens + cand_tokens
    title_cand_ids = tokenizer.convert_tokens_to_ids(title_cand_tokens)

    padding = [0] * (max_seq_length - len(title_cand_ids))
    title_cand_ids += padding
    assert len(title_cand_ids) == max_seq_length

    return {"tokens": title_cand_tokens, "ids": title_cand_ids}


def process_mention_data(
    samples,
    tokenizer,
    max_context_length,
    max_cand_length,
    silent,
    mention_key="mention",
    context_key="context",
    entity_key="entity",
    ent_start_token=ENT_START_TAG,
    ent_end_token=ENT_END_TAG,
    just_mention=False,
    debug=False,
    logger=None,
):
    processed_samples = []

    if debug:
        import random

        random.seed(42)
        samples = random.sample(samples, 300)

    if silent:
        iter_ = samples
    else:
        iter_ = tqdm(samples, dynamic_ncols=True)

    for idx, sample in enumerate(iter_):
        mention_context_tokens = get_context_representation(
            sample,
            tokenizer,
            max_context_length,
            mention_key,
            context_key,
            ent_start_token,
            ent_end_token,
        )
        if not just_mention:
            entity_info = sample.get(entity_key, None)
            qid, title, desc = entity_text_modeling(entity_info)

            title_desc_tokens = get_candidate_representation(
                desc,
                tokenizer,
                max_cand_length,
                title,
            )
            label_idx = int(sample["label_id"])

            record = {
                "mention": mention_context_tokens,
                "cand": title_desc_tokens,
                "label_idx": [label_idx],
            }
        else:
            record = {"mention": mention_context_tokens}

        processed_samples.append(record)

    if debug and logger and not just_mention:
        logger.info("====Processed samples: ====")
        for sample in processed_samples[:5]:
            logger.info("Mention tokens : " + " ".join(sample["mention"]["tokens"]))
            logger.info("Mention ids : " + " ".join([str(v) for v in sample["mention"]["ids"]]))
            logger.info("Context tokens : " + " ".join(sample["cand"]["tokens"]))
            logger.info("Context ids : " + " ".join([str(v) for v in sample["cand"]["ids"]]))
            logger.info("Label_id : %d" % sample["label_idx"][0])

    mention_vecs = torch.tensor(
        select_field(processed_samples, "mention", "ids"),
        dtype=torch.long,
    )
    if not just_mention:
        cand_vecs = torch.tensor(
            select_field(processed_samples, "cand", "ids"),
            dtype=torch.long,
        )
        label_idx = torch.tensor(
            select_field(processed_samples, "label_idx"),
            dtype=torch.long,
        )
        data = {
            "mention_vecs": mention_vecs,
            "cand_vecs": cand_vecs,
            "label_idx": label_idx,
        }
    else:
        data = {"mention_vecs": mention_vecs}

    if not just_mention:
        tensor_data = TensorDataset(mention_vecs, cand_vecs, label_idx)
    else:
        tensor_data = TensorDataset(mention_vecs)

    del processed_samples
    gc.collect()

    return data, tensor_data
