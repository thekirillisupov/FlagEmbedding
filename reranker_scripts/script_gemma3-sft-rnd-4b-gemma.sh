export WANDB_MODE=disabled

train_data_msmarco="/home/jovyan/isupov/reranker/data/rnd_retrieval_data/msmarco.jsonl"
train_data_nq="/home/jovyan/isupov/reranker/data/rnd_retrieval_data/nq.jsonl"


# miracle
train_data_ru_miracl="/home/jovyan/isupov/reranker/data/rnd_retrieval_data/ru_miracl.jsonl"
train_data_en_miracl="/home/jovyan/isupov/reranker/data/rnd_retrieval_data/en_miracl.jsonl"
# mr tydi
train_data_ru_mr_tydi="/home/jovyan/isupov/reranker/data/rnd_retrieval_data/ru_mr_tydi.jsonl"
train_data_en_mr_tydi="/home/jovyan/isupov/reranker/data/rnd_retrieval_data/en_mr_tydi.jsonl"
train_data_taiga_fontanka="/home/jovyan/isupov/reranker/data/rnd_retrieval_data/taiga_fontanka.jsonl"
train_data_point_wise="/home/jovyan/isupov/reranker/data/qwen_sft/point_wise_samples_v1_13_09_13.json"
# mldr data
train_data_en_mldr="/home/jovyan/isupov/FlagEmbedding/mldr_en_train.query_pos_neg.jsonl"
train_data_ru_mldr="/home/jovyan/isupov/FlagEmbedding/mldr_ru_train.query_pos_neg.jsonl"

# set large epochs and small batch size for testing
num_train_epochs=1
per_device_train_batch_size=1
gradient_accumulation_steps=1
train_group_size=8
# actual batch size is per_device_train_batch_size * train_group_size * num_gpus * gradient_accumulation_steps 

# set num_gpus to 2 for testing
num_gpus=8

if [ -z "$HF_HUB_CACHE" ]; then
    export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
fi

model_args="\
    --model_format gemma3 \
    --model_name_or_path google/gemma-3-4b-it \
    --cache_dir $HF_HUB_CACHE \
    --use_lora True \
    --use_flash_attn True \
    --lora_rank 32 \
    --lora_alpha 64 \
    --target_modules q_proj k_proj v_proj o_proj \
    --save_merged_lora_model False \
    --model_type decoder \
"
# TODO set to 16 

data_args="\
    --train_data ${train_data_msmarco} ${train_data_en_miracl} ${train_data_en_mr_tydi} ${train_data_nq} ${train_data_ru_miracl} ${train_data_ru_mr_tydi} ${train_data_taiga_fontanka} \
    --cache_path ~/.cache \
    --train_group_size $train_group_size \
    --query_max_len 128 \
    --passage_max_len 1000 \
    --pad_to_multiple_of 8 \
    --knowledge_distillation False \
    --query_instruction_for_rerank 'A: ' \
    --query_instruction_format '{}{}' \
    --passage_instruction_for_rerank 'B: ' \
    --passage_instruction_format '{}{}' \
"


#     --deepspeed ../../ds_stage0.json \
#     --gradient_checkpointing \
training_args="\
    --logit_calculation_type margin_score \
    --output_dir ./output/decoder_gemma_3_rnd_4b_gemma \
    --overwrite_output_dir \
    --gradient_checkpointing \
    --learning_rate 1e-4 \
    --max_grad_norm 1.0 \
    --bf16 \
    --num_train_epochs $num_train_epochs \
    --per_device_train_batch_size $per_device_train_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --logging_steps 1000 \
    --save_strategy epoch \
"

cmd="torchrun --nproc_per_node $num_gpus \
    -m FlagEmbedding.finetune.reranker.decoder_only.base \
    $model_args \
    $data_args \
    $training_args \
"

echo $cmd
eval $cmd