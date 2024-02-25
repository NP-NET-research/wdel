export CUDA_VISIBLE_DEVICES=0

echo "eval biencoder"
python src/candidate_generation/eval.py \
    --model_name "/home/xuzf/models/bert-base-multilingual-uncased" \
    --output_path output/cg \
    --data_path data/hansel \
    --entity_dict_path KB/wikidata-2023-12-22/tmp/wk_brief_info \
    --mode "hansel-test-fs,hansel-test-zs" \
    --dim 128 \
    --path_to_model output/cg/train_retriever_be@01-27-12:40/pytorch_model.bin \
    --cand_pool_path output/cg/train_retriever_be@01-27-12:40/candidate_pool \
    --cand_encode_path output/cg/train_retriever_be@01-27-12:40/wk_info_all_emb.t7 \
    --save_topk_result \
    --encode_batch_size 2000 \
    --eval_batch_size 64 \
    --max_context_length 128 \
    --max_cand_length 128 \
    --top_k 10
    # --cand_encode_path output/cg/train_retriever_be@01-23-12:56/candidate_emb \
    # --cand_encode_path data/hansel/wikidata/vector_index/01-23-12:56/hansel_wk_emb.t7 \