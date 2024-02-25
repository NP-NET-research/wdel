export CUDA_VISIBLE_DEVICES=0

python src/entity_disambiguation/train.py \
    --train_args_json src/entity_disambiguation/qwen_7B_LoRA.json \
    --model_name_or_path /home/xuzf/models/qwen/Qwen-7B-Chat \
    --train_data_path data/hansel/hansel-train-1M-h10.jsonl \
    --eval_data_path data/hansel/hansel-dev-h10.jsonl \
    --max_context_length 128 \
    --max_desc_length 256 \
    --max_train_samples 3000000 \
    --max_eval_samples 20000 \
    --neg_sample_num 3 \
    --lora_rank 128 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --compute_dtype fp16 \
    --peft_type "lora" \
