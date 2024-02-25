# set cuda env
export CUDA_VISIBLE_DEVICES=0

echo "train retriever"
python src/candidate_generation/train.py \
    --model_name "/home/xuzf/models/bert-base-multilingual-uncased" \
    --data_path "data/hansel" \
    --print_interval 50 \
    --eval_interval 250 \
    --dim 128 \
    --path_to_model "output/cg/train_retriever_be@01-26-01:04/pytorch_model.bin" \
    --learning_rate 1e-5 \
    --train_batch_size 256 \
    --num_train_epochs 1 \
    --train_steps_per_epoch 25000 \
    --eval_batch_size 1024 \
    --output_path output/cg \
    --warmup_proportion 0.1 \
    --shuffle True \
    --max_context_length 128 \
    --max_cand_length 128 \
    --use_wandb \
    --comment "train in batch on the hansel dataset"
    