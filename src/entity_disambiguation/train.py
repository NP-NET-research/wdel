#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =============================================================================
# @Author  :  xuzf
# @Contact :  xuzhengfei-email@qq.com
# @Create  :  2023-08-20 18:56:55
# @Update  :  2023-11-26 03:40:44
# @Desc    :  None
# =============================================================================


import os
import time
import json
import jsonlines
import argparse
from typing import List, Dict, Optional
import sys

root_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(root_dir)

from src.candidate_generation.tokenization import entity_text_modeling

import torch

from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    BitsAndBytesConfig,
    set_seed,
)
from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    set_peft_model_state_dict,
)

from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

_compute_dtype_map = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def parse_args():
    parser = argparse.ArgumentParser(description="LLm4EL ChatGLM-6B QLoRA")
    parser.add_argument("--train_args_json", type=str, required=True, help="TrainingArguments的json文件")
    parser.add_argument("--model_name_or_path", type=str, default="THUDM/chatglm2-6b", help="模型id或local path")
    parser.add_argument(
        "--peft_type", type=str, default="lora", choices=["lora", "prompt_tuning", "prefix_tuning"], help="微调方法"
    )
    parser.add_argument("--train_data_path", type=str, required=True, help="训练数据路径")
    parser.add_argument("--eval_data_path", type=str, default=None, help="验证数据路径")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_context_length", type=int, default=256, help="context的最大长度")
    parser.add_argument("--max_desc_length", type=int, default=256, help="entity description的最大长度")
    parser.add_argument("--lora_rank", type=int, default=4, help="lora rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="lora_alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="lora dropout")
    parser.add_argument("--max_train_samples", type=int, default=1000000, help="训练样本数")
    parser.add_argument("--max_eval_samples", type=int, default=100000, help="验证样本数")
    parser.add_argument("--neg_sample_num", type=int, default=3, help="负采样数")
    # parser.add_argument("--num_virtual_tokens", type=int, default=10, help="prefix virtual prompt length")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="恢复训练的checkpoint路径")
    parser.add_argument("--prompt_text", type=str, default="", help="统一添加在所有数据前的指令文本")
    parser.add_argument("--compute_dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"], help="计算数据类型")
    return parser.parse_args()


def tokenize_func_chatglm2(example, tokenizer, ignore_label_id=-100):
    """单样本tokenize处理"""
    question = "[Round 1]\n\n问：{}\n\n答：".format(example["inputs"])

    answer = example["outputs"]
    q_ids = tokenizer.encode(text=question, add_special_tokens=False)
    # a_ids = tokenizer.encode(text=answer, add_special_tokens=False)
    a_ids = [tokenizer.convert_tokens_to_ids(answer)]

    input_ids = tokenizer.build_inputs_with_special_tokens(q_ids, a_ids)
    question_length = len(q_ids) + 2  # chatglm1 - gmask, bos, chatglm2 - gmask, sop
    labels = [ignore_label_id] * question_length + input_ids[question_length:]
    return {"input_ids": input_ids, "labels": labels}


def tokenize_func_chatglm3(example, tokenizer, ignore_label_id=-100):
    """单样本tokenize处理"""
    question = example["inputs"] + "\n"
    q_ids = tokenizer.build_single_message("user", "", message=question)
    q_ids.extend([tokenizer.get_command("<|assistant|>")])
    answer = example["outputs"]
    a_ids = [tokenizer.convert_tokens_to_ids(answer)]
    input_ids = tokenizer.build_inputs_with_special_tokens(q_ids, a_ids)
    question_length = len(q_ids) + 2  # chatglm1 - gmask, bos, chatglm2 - gmask, sop, chatglm3 - gmask, sop
    labels = [ignore_label_id] * question_length + input_ids[question_length:]
    return {"input_ids": input_ids, "labels": labels}


def tokenize_func_qwen(
    example,
    tokenizer,
):
    input_ids, labels = [], []

    instruction_text = "\n".join(
        [
            "<|im_start|>system",
            "You are a helpful assistant.<|im_end|>",
            "<|im_start|>user",
            example["inputs"] + "<|im_end|>\n",
        ]
    )
    instruction = tokenizer(instruction_text, add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    response_text = "<|im_start|>assistant\n" + example["outputs"] + "<|im_end|>\n"
    response = tokenizer(response_text, add_special_tokens=False)

    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    labels = (
        [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    )  # Qwen的特殊构造就是这样的
    return {"input_ids": input_ids, "labels": labels}


def get_dataset(
    data_path,
    tokenizer,
    prompt_text,
    max_context_length,
    max_desc_length,
    neg_num=3,
    max_train_samples=1000000,
    max_eval_samples=100000,
    model_name="qwen",
):
    if "chatglm2" in model_name:
        tokenize_func = tokenize_func_chatglm2
    if "chatglm3" in model_name:
        tokenize_func = tokenize_func_chatglm3
    if "qwen" in model_name:
        tokenize_func = tokenize_func_qwen

    if "train" in data_path:
        neg_num = neg_num
    elif "dev" in data_path:
        neg_num = 9
    dataset = []
    with open(data_path, "r") as f:
        for line in f.readlines():
            item = json.loads(line.strip())
            label = item["label"]
            if label == -1:
                continue
            mention = item["mention"]
            mention_length = len(mention) + 4
            context_left_quota = (max_context_length - mention_length) // 2
            context_right_quota = max_context_length - mention_length - context_left_quota
            context = (
                item["context_left"][-context_left_quota:] + f"<<{mention}>>" + item["context_right"][:context_right_quota]
            )
            gold_cand = item["candidates"][label]
            neg_cands = item["candidates"][:label] + item["candidates"][label + 1 :]
            _, gold_cand_label, gold_cand_desc = entity_text_modeling(gold_cand)
            gold_inputs = (
                prompt_text
                + f"\n【文本】：{context}\n【指称】：{mention}\n【实体】：{gold_cand_label}\n【实体描述】：{gold_cand_desc[:max_desc_length]}"
            ).strip()
            gold_outputs = "是"
            example = {"inputs": gold_inputs, "outputs": gold_outputs}
            dataset.append(tokenize_func(example, tokenizer))
            # 保留1个错误的候选实体
            for id, cand in enumerate(neg_cands[:neg_num]):
                _, cand_label, cand_desc = entity_text_modeling(cand)
                inputs = (
                    prompt_text
                    + f"\n【文本】：{context}\n【指称】：{mention}\n【实体】：{cand_label}\n【实体描述】：{cand_desc[:max_desc_length]}"
                ).strip()
                outputs = "否"
                example = {"inputs": inputs, "outputs": outputs}
                dataset.append(tokenize_func(example, tokenizer))

            if "train" in data_path and len(dataset) >= max_train_samples:
                break
            elif "dev" in data_path and len(dataset) >= max_eval_samples:
                break
    return dataset


class DataCollatorForLLM:
    def __init__(self, pad_token_id: int, max_length: int = 1024, ignore_label_id: int = -100):
        self.pad_token_id = pad_token_id
        self.ignore_label_id = ignore_label_id
        self.max_length = max_length

    def __call__(self, batch_data: List[Dict[str, List]]) -> Dict[str, torch.Tensor]:
        """根据batch最大长度做padding"""
        len_list = [len(d["input_ids"]) for d in batch_data]
        batch_max_len = max(len_list)
        input_ids, labels = [], []
        for len_of_d, d in sorted(zip(len_list, batch_data), key=lambda x: -x[0]):
            pad_len = batch_max_len - len_of_d
            ids = d["input_ids"] + [self.pad_token_id] * pad_len
            label = d["labels"] + [self.ignore_label_id] * pad_len
            if batch_max_len > self.max_length:
                ids = ids[: self.max_length]
                label = label[: self.max_length]
            input_ids.append(torch.LongTensor(ids))
            labels.append(torch.LongTensor(label))
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        return {"input_ids": input_ids, "labels": labels}


class PeftTrainer(Trainer):
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """只保存adapter"""
        if output_dir is None:
            output_dir = self.args.output_dir
        self.model.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


def train(global_args):
    hf_parser = HfArgumentParser(TrainingArguments)
    (hf_train_args,) = hf_parser.parse_json_file(json_file=global_args.train_args_json)
    model_output_path = os.path.join(
        hf_train_args.output_dir, "train_lora_reranker@%s" % (time.strftime("%m-%d-%H:%M", time.localtime()))
    )
    hf_train_args.output_dir = model_output_path

    set_seed(global_args.seed)
    hf_train_args.seed = global_args.seed
    model_max_length = (global_args.max_context_length + global_args.max_desc_length) * 4

    if global_args.peft_type == "lora":
        # global_args.prompt_text = '判断下面文本中的指称“{}”是否指代下面的实体“{}”。如果提及指代给定实体则输出"是"，否则输出"否"。'
        global_args.prompt_text = '判断文本中用 "<<" 和 ">>" 标记起始和终止位置的指称是否指代给定的实体。如果提及指代给定实体则输出"是"，否则输出"否"。'

        # Quantization
        # q_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_compute_dtype=_compute_dtype_map[global_args.compute_dtype],
        # )
        # model = AutoModel.from_pretrained(
        #     global_args.model_name_or_path, quantization_config=q_config, device_map="cuda:0", trust_remote_code=True
        # )
        # model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        if "glm" in global_args.model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(global_args.model_name_or_path, trust_remote_code=True)
            model = AutoModel.from_pretrained(global_args.model_name_or_path, trust_remote_code=True).to("cuda")
            target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING["chatglm"]
        elif "qwen" in global_args.model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(global_args.model_name_or_path, use_fast=False, trust_remote_code=True)
            tokenizer.pad_token_id = tokenizer.eod_id  # Qwen中eod_id和pad_token_id是一样的，但需要指定一下
            model = AutoModelForCausalLM.from_pretrained(
                global_args.model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16
            ).to("cuda")
            if hf_train_args.gradient_checkpointing:
                model.enable_input_require_grads()
            target_modules = ["c_attn", "c_proj", "w1", "w2"]

        # LoRA
        lora_config = LoraConfig(
            r=global_args.lora_rank,
            lora_alpha=global_args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=global_args.lora_dropout,
            bias="none",
            inference_mode=False,
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)

    # 加载checkpoint
    resume_from_checkpoint = global_args.resume_from_checkpoint
    if resume_from_checkpoint is not None:
        checkpoint_name = os.path.join(resume_from_checkpoint, "pytorch_model.bin")
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(resume_from_checkpoint, "adapter_model.safetensors")
            resume_from_checkpoint = False

        if os.path.exists(checkpoint_name):
            if global_args.peft_type == "lora":
                if checkpoint_name.endswith("adapter_model.safetensors"):
                    from safetensors import safe_open
                    adapters_weights = {}
                    with safe_open(checkpoint_name, framework="pt", device='cpu') as f:
                        for k in f.keys():
                            adapters_weights[k] = f.get_tensor(k)
                else:
                    adapters_weights = torch.load(checkpoint_name)
                    set_peft_model_state_dict(model, adapters_weights)
                    model.print_trainable_parameters()

    # data
    train_dataset = get_dataset(
        global_args.train_data_path,
        tokenizer,
        global_args.prompt_text,
        global_args.max_context_length,
        global_args.max_desc_length,
        global_args.neg_sample_num,
        global_args.max_train_samples,
        global_args.max_eval_samples,
        global_args.model_name_or_path,
    )
    with jsonlines.open("data/hansel/rerank_qwen_train_h4_128_256.jsonl", "w") as f:
        for line in train_dataset:
            f.write(line)
    # train_dataset = []
    # with jsonlines.open("data/hansel/rerank_qwen_train_h4_128_256.jsonl", "r") as f:
    #     for line in f:
    #         train_dataset.append(line)
    eval_dataset = None
    if global_args.eval_data_path:
        eval_dataset = get_dataset(
            global_args.eval_data_path,
            tokenizer,
            global_args.prompt_text,
            global_args.max_context_length,
            global_args.max_desc_length,
            global_args.neg_sample_num,
            global_args.max_train_samples,
            global_args.max_eval_samples,
            global_args.model_name_or_path,
        )
        with jsonlines.open("data/hansel/rerank_qwen_dev_h4_128_256.jsonl", "w") as f:
            for line in eval_dataset:
                f.write(line)
        # eval_dataset = []
        # with jsonlines.open("data/hansel/rerank_qwen_dev_h4_128_256.jsonl", "r") as f:
        #     for line in f:
        #         eval_dataset.append(line)
        

    data_collator = DataCollatorForLLM(pad_token_id=tokenizer.pad_token_id, max_length=model_max_length)

    # train
    trainer = PeftTrainer(
        model=model,
        args=hf_train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.model.save_pretrained(hf_train_args.output_dir)


if __name__ == "__main__":
    args = parse_args()
    train(args)
