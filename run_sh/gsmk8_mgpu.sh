#! /bin/bash
# runs on multiple gpus
export HF_HOME="/home/rares/LLaMA-Factory/large_data/cache_hugg"

pretraine_model=/home/rares/LLaMA-Factory/large_data/qwen2_vl-7b/full/sft
# pretraine_model=Qwen/Qwen2.5-7B
# lm_eval --model hf \
#     --model_args pretrained=$pretraine_model,dtype="float",max_length=2048 \
#     --tasks gsm8k_cot \
#     --device cuda:0 \
#     --batch_size 64 \
#     #--limit 1000

accelerate launch -m lm_eval --model hf \
    --model_args pretrained=$pretraine_model,dtype="float",max_length=2048 \
    --tasks gsm8k_cot \
    --batch_size 16