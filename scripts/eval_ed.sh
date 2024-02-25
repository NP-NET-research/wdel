# !/bin/bash
export CUDA_VISIBLE_DEVICES=0

python src/entity_disambiguation/eval.py \
    --peft_model_path output/lora/train_lora_reranker@02-11-13:50 \
    --data_path data/hansel/hansel-test-fs-h10.jsonl,data/hansel/hansel-test-zs-h10.jsonl