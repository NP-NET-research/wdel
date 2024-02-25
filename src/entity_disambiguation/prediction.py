#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =============================================================================
# @Author  :  xuzf
# @Contact :  xuzhengfei-email@qq.com
# @Create  :  2023-08-23 10:46:34
# @Update  :  2023-11-06 09:30:31
# @Desc    :  None
# =============================================================================

import torch
from tqdm import trange
from typing import List, Tuple, Dict
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM

from peft import PeftModel, PeftConfig

import sys

sys.path.append("/home/xuzf/models/qwen/Qwen-7B-Chat")

from qwen_generation_utils import make_context
from utils.io import torch_gc


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


# class InvalidScoreLogitsProcessor(LogitsProcessor):
#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
#         if torch.isnan(scores).any() or torch.isinf(scores).any():
#             scores.zero_()
#             scores[..., 5] = 5e4
#         return scores


@torch.inference_mode()
def inference(model, tokenizer, querys: List[Dict], batch_size: int = 2, silent: bool = True, **kwargs):
    """
    输入：
    querys: List[Dict], example:
    [
        {
            "inputs": "判断文本中用 '<<' 和 '>>' 标记起始和终止位置的提及是否指代给定的实体。如果提及指代给定实体则输出'是'，否则输出'否'。",
            "query_id": "0-0",
        },
        ...
    ]
    输出：
    [
        {
            "query": "0",
            "outputs": ["否"，"否", "是"],
            "scores": [0.0， 0.0, 1.0],
        },
       ...
    ]
    """
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
    batch_tokenize_func = batch_tokenize_func_qwen
    with torch.no_grad():
        model.eval()

        preds = {}
        for start_index in trange(0, len(querys), batch_size, desc="LoRA infer", disable=silent, dynamic_ncols=True):
            # 处理batch数据
            query_data = querys[start_index : start_index + batch_size]
            query_text_list = [query["inputs"] for query in query_data]
            query_id_list = [query["query_id"] for query in query_data]
            querys_ids = batch_tokenize_func(query_text_list, tokenizer).to(model.device)
            # 推理
            outputs = model.generate(**querys_ids, **gen_kwargs)
            scores, gen_ids = outputs.scores[0].max(dim=-1)
            gen_list = tokenizer.batch_decode(gen_ids.tolist())
            scores = scores.cpu().tolist()
            # 保存结果
            for query_id, gen, logit in zip(query_id_list, gen_list, scores):
                query_id = query_id.split("-")
                sample_idx = int(query_id[0])
                cand_idx = int(query_id[1])
                preds[sample_idx] = preds.get(sample_idx, {})
                preds[sample_idx][cand_idx] = {"gen": gen, "logit": logit}
        torch_gc()
    response = []
    for sample_idx, cands_preds in preds.items():
        outputs = []
        scores = []
        for cand_idx, pred in cands_preds.items():
            out_token, out_score = pred["gen"], pred["logit"]
            outputs.append(out_token)
            scores.append(out_score)
        # 归一化，消除量纲
        min_score = min(scores)
        maxmin_score = max(scores) - min_score
        if maxmin_score != 0:
            scores = [
                1 + (1 if outputs[i] == "是" else -1) * (scores[i] - min_score) / maxmin_score for i in range(len(scores))
            ]
            # softmax 计算概率
            lora_scores = torch.softmax(torch.tensor(scores, dtype=torch.float32), dim=0).tolist()
        else:
            lora_scores = [1 / len(scores)] * len(scores)

        response.append({"query_id": sample_idx, "outputs": outputs, "scores": lora_scores})

    return response


def main():
    peft_model_path = "saved_files/chatGLM_6B_QLoRA_t32"
    model, tokenizer = load_lora_model(peft_model_path)
    query = [
        {
            "inputs": '判断文本中用 "<<" 和 ">>" 标记起始和终止位置的提及是否指代给定的实体。如果提及指代给定实体则输出"是"，否则输出"否"。\n文本：记》有点类似而已，并不能确证它们一定起源于印度，很可能只是巧合。Section::::话本、戏曲.在唐朝后期和五代时期的许多记载中已经出现了西行取经的故事。现存敦煌石窟的玄奘取经壁画，大约作于西夏初年，已经出现持棒猴行者形象；南宋刊印的话本《<<大唐三藏取经诗话>>》，已经有猴行者化作白衣秀士，自称“花果山紫云洞八万四千铜头铁额猕猴王”和“深沙神”。宋元南戏有《陈光蕊江流和尚》，吴昌龄作杂剧《唐三藏西天取经》已经有师徒四众，元朝杨景贤作杂剧《西游记》；元末明初的杂剧《二郎神锁齐天大圣》和《西游\n提及：大唐三藏取经诗话\n实体名称：大唐三藏取经\n实体描述：《大唐三藏取经诗话》为一话本，作者不详，三卷十七段，为猴行者助三藏取经的故事，无猪八戒，有深沙神（疑为沙僧原型），为明代小说《西游记》的来源之一。  此话本在中国原已失传，近代时才在日本重新发现。',
            "query_id": "0-0",
        },
        {
            "inputs": '判断文本中用 "<<" 和 ">>" 标记起始和终止位置的提及是否指代给定的实体。如果提及指代给定实体则输出"是"，否则输出"否"。\n文本：记》有点类似而已，并不能确证它们一定起源于印度，很可能只是巧合。Section::::话本、戏曲.在唐朝后期和五代时期的许多记载中已经出现了西行取经的故事。现存敦煌石窟的玄奘取经壁画，大约作于西夏初年，已经出现持棒猴行者形象；南宋刊印的话本《<<大唐三藏取经诗话>>》，已经有猴行者化作白衣秀士，自称“花果山紫云洞八万四千铜头铁额猕猴王”和“深沙神”。宋元南戏有《陈光蕊江流和尚》，吴昌龄作杂剧《唐三藏西天取经》已经有师徒四众，元朝杨景贤作杂剧《西游记》；元末明初的杂剧《二郎神锁齐天大圣》和《西游\n提及：大唐三藏取经诗话\n实体名称：大唐三藏取经\n实体描述：《大唐三藏取经诗话》为一话本，作者不详，三卷十七段，为猴行者助三藏取经的故事，无猪八戒，有深沙神（疑为沙僧原型），为明代小说《西游记》的来源之一。  此话本在中国原已失传，近代时才在日本重新发现。',
            "query_id": "0-1",
        },
    ]
    query = query.strip()
    inference(model, tokenizer, query)


if __name__ == "__main__":
    main()
