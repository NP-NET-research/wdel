#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

echo "eval biencoder"

file_index_list=(0 10000000 20000000 30000000 40000000 50000000 60000000 70000000 80000000 90000000 100000000)
for file_index in "${file_index_list[@]}"; do
    python src/candidate_generation/eval.py \
        --model_name "/home/xuzf/models/bert-base-multilingual-uncased" \
        --output_path output/cg \
        --data_path data/hansel \
        --entity_dict_path KB/wikidata-2023-12-22/tmp/wk_brief_info/wk_info_${file_index}.jsonl \
        --mode "encode_candidate" \
        --path_to_model output/cg/train_retriever_be@01-27-12:40/pytorch_model.bin \
        --dim 128 \
        --cand_pool_path output/cg/train_retriever_be@01-27-12:40/candidate_pool/wk_info_${file_index}_pool.t7 \
        --cand_encode_path output/cg/train_retriever_be@01-27-12:40/candidate_emb/wk_info_${file_index}_emb.t7 \
        --save_topk_result \
        --encode_batch_size 2000 \
        --eval_batch_size 1024 \
        --max_context_length 128 \
        --max_cand_length 128 \
        --top_k 10
done