#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =============================================================================
# @Author  :  xuzf
# @Contact :  xuzhengfei-email@qq.com
# @Create  :  2023-08-24 16:49:09
# @Update  :  2023-10-28 19:03:29
# @Desc    :  None
# =============================================================================

import os
import argparse
import json
import torch
from tqdm import tqdm, trange
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

import sys

root_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(root_dir)
sys.path.append("/home/xuzf/models/qwen/Qwen-7B-Chat")

from qwen_generation_utils import make_context
from src.candidate_generation.tokenization import entity_text_modeling
from utils.io import torch_gc

def load_data(data_path):
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores


def load_lora_model(peft_model_path: str, compute_dtype: str = "fp32", device: str = "cuda"):
    config = PeftConfig.from_pretrained(peft_model_path)
    # q_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_compute_dtype=_compute_dtype_map[compute_dtype],
    # )
    if "qwen" in config.base_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model_name_or_path,
            pad_token="<|extra_0|>",
            eos_token="<|endoftext|>",
            padding_side="left",
            trust_remote_code=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            # quantization_config=q_config,
            pad_token_id=tokenizer.pad_token_id,
            trust_remote_code=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(config.base_model_name_or_path, trust_remote_code=True)

    model = PeftModel.from_pretrained(base_model, peft_model_path)
    model = model.merge_and_unload()
    model = model.to(device)
    return model, tokenizer, config.base_model_name_or_path


def batch_tokenize_func_chatglm2(batch_text, tokenizer):
    querys = ["[Round 1]\n\n问：{}\n\n答：".format(query.strip()) for query in batch_text]
    return tokenizer(querys, padding=True, return_tensors="pt")


def batch_tokenize_func_chatglm3(batch_text, tokenizer):
    batch_inputs = []
    for query in batch_text:
        query = query.strip() + "\n"
        query = tokenizer.build_single_message("user", "", message=query) + [tokenizer.get_command("<|assistant|>")]
        batch_inputs.append(query)
    batch_inputs = tokenizer.batch_encode_plus(batch_inputs, padding="longest", return_tensors="pt", is_split_into_words=True)
    return batch_inputs


def batch_tokenize_func_qwen(batch_text, tokenizer):
    batch_inputs = []
    for query in batch_text:
        query = query.strip()
        formar_query, _ = make_context(
            tokenizer,
            query,
            system="You are a helpful assistant.",
            max_window_size=6144,
            chat_format="chatml",
        )
        batch_inputs.append(formar_query)

    batch_input_ids = tokenizer(batch_inputs, padding="longest", return_tensors="pt")
    return batch_input_ids


def eval_el_lora(data, model, tokenizer, prompt_text, batch_size, max_context_length, max_desc_length, model_name, **kwargs):
    # 推理参数
    logits_processor = LogitsProcessorList()
    logits_processor.append(InvalidScoreLogitsProcessor())
    # 处理数据生成query
    eval_data = []
    for i, item in enumerate(data):
        # querys = []
        mention = item["mention"]
        mention_length = len(mention) + 4
        context_left_quota = max_context_length - mention_length // 2
        context_right_quota = max_context_length - mention_length - context_left_quota
        context = item["context_left"][-context_left_quota:] + f"<<{mention}>>" + item["context_right"][:context_right_quota]
        label = item["label"]
        for j, cand in enumerate(item["candidates"]):
            _, cand_label, cand_desc = entity_text_modeling(cand)
            inputs = (
                prompt_text
                + f"\n【文本】：{context}\n【指称】：{mention}\n【实体】：{cand_label}\n【实体描述】：{cand_desc[:max_desc_length]}"
            )
            eval_data.append({"query_id": f"{i}-{j}", "inputs": inputs})
            # querys.append(inputs)
        # eval_data.append({"querys": querys, "label": label})
    # 生成预测结果
    gen_kwargs = {
        "max_length": 8192,
        "num_beams": 1,
        "do_sample": False,
        "logits_processor": logits_processor,
        "return_dict_in_generate": True,
        "repetition_penalty": 10.0,
        "output_scores": True,
        **kwargs,
    }
    # "top_p": 0.8,
    # "temperature": 0.8,
    if "chatglm2" in model_name:
        batch_tokenize_func = batch_tokenize_func_chatglm2
    if "chatglm3" in model_name:
        batch_tokenize_func = batch_tokenize_func_chatglm3
    if "qwen" in model_name:
        batch_tokenize_func = batch_tokenize_func_qwen
        gen_kwargs = {
            "chat_format": "chatml",
            "eos_token_id": 151643,
            "pad_token_id": 151643,
            "max_window_size": 6144,
            "max_new_tokens": 512,
            "do_sample": False,
            "repetition_penalty": 1.1,
            "transformers_version": "4.31.0",
            "return_dict_in_generate": True,  # 新增
            "output_scores": True,  # 新增
            **kwargs,
        }
        # "top_k": 0,
        # "top_p": 0.8,

    with torch.no_grad():
        model.eval()
        # res_list = []
        # query_dataset = []

        # for idx, item in enumerate(tqdm(eval_data)):
        for start_index in trange(0, len(eval_data), batch_size, desc="LoRA infer", disable=False, dynamic_ncols=True):
            query_data = eval_data[start_index : start_index + batch_size]
            querys = [query["inputs"] for query in query_data]
            query_id_list = [query["query_id"] for query in query_data]
            querys_ids = batch_tokenize_func(querys, tokenizer).to(model.device)
            outputs = model.generate(**querys_ids, **gen_kwargs)
            scores, gen_ids = outputs.scores[0].max(dim=-1)
            gen_list = tokenizer.batch_decode(gen_ids.tolist())
            scores = scores.cpu().tolist()
            for query_id, gen, logit in zip(query_id_list, gen_list, scores):
                query_id = query_id.split("-")
                sample_idx = int(query_id[0])
                cand_idx = int(query_id[1])
                data[sample_idx]["candidates"][cand_idx]["lora_pred"] = (gen, logit)
            torch_gc()
            # if start_index % (2 * batch_size) == 0:
            #     torch_gc()
    preds_list = []
    for idx, sample in enumerate(data):
        preds = []
        scores = []
        for cand in sample["candidates"]:
            pred, score = cand["lora_pred"]
            preds.append(pred)
            scores.append(score)
        # 归一化，消除量纲
        min_score = min(scores)
        maxmin_score = max(scores) - min_score
        if maxmin_score != 0:
            scores = [1 + (1 if preds[i] == "是" else -1) * (scores[i] - min_score) / maxmin_score for i in range(len(scores))]
            # softmax 计算概率
            lora_scores = torch.softmax(torch.tensor(scores, dtype=torch.float32), dim=0).tolist()
        else:
            lora_scores = [1 / len(scores)] * len(scores)
        data[idx]["lora_score"] = lora_scores
        # 两步概率相乘
        # if with_confidence:
        #     cg_scores = sample["cg_score"]
        #     scores = [0.1 * cg_score + icl_score for cg_score, icl_score in zip(cg_scores, icl_scores)]
        # else:
        #     scores = icl_scores
        final_pred = scores.index(max(scores))
        preds_list.append(final_pred)
        data[idx]["gen"] = preds
        data[idx]["pred"] = final_pred

    # 计算accuarcy
    labels = [sample["label"] for sample in data]
    correct = 0
    for pred, label in zip(preds_list, labels):
        if pred == label:
            correct += 1
    acc = correct / len(labels)
    # new_preds = preds_list
    return acc, data


def main(args):
    peft_model_path = args.peft_model_path
    model, tokenizer, model_name = load_lora_model(peft_model_path, compute_dtype="fp16")
    data_paths = args.data_path
    data_paths = data_paths.split(",")
    for data_path in data_paths:
        # prompt_text = '判断下面文本中的指称“{}”是否指代下面的实体“{}”。如果提及指代给定实体则输出"是"，否则输出"否"。'
        prompt_text = '判断文本中用 "<<" 和 ">>" 标记起始和终止位置的指称是否指代给定的实体。如果提及指代给定实体则输出"是"，否则输出"否"。'
        data = load_data(data_path)
        batch_size = 10
        accuracy, data = eval_el_lora(data, model, tokenizer, prompt_text, batch_size, 128, 256, model_name=model_name)
        print(accuracy)

        result_fpath = os.path.join(peft_model_path, data_path.split("/")[-1].replace(".jsonl", "-pred-result.jsonl"))
        with open(result_fpath, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        with open(result_fpath.replace("-pred-result.jsonl", "-pred-accuracy.txt"), "w", encoding="utf-8") as f:
            f.write("Accuracy: {:.5f}".format(accuracy))


if __name__ == "__main__":
    # 编写 args parser 部分，要求有base_model_path, peft_model_path, data_path, prompt_text
    parser = argparse.ArgumentParser(description="Evaluate the LoRA model on the entity linking task.")
    parser.add_argument("--peft_model_path", type=str, help="Path to the PEFT model")
    parser.add_argument("--data_path", type=str, help="Path to the test data")

    args = parser.parse_args()

    main(args)

    # main(1)
