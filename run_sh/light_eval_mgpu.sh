#!/bin/bash
export HF_HOME="/home/rares/LLaMA-Factory/large_data/cache_hugg"

MODEL=Qwen/Qwen2.5-7B
#knoveleng/Open-RS3
MODEL_ARGS="pretrained=$MODEL,data_parallel_size=4,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=large_data/evals/$MODEL

export VLLM_WORKER_MULTIPROC_METHOD=spawn
# Example: AIME 2024
TASK=aime24
lighteval vllm "$MODEL_ARGS" "custom|$TASK|0|0" \
  --custom-tasks src/evaluation/math_evaluate.py \
  --use-chat-template \
  --output-dir "$OUTPUT_DIR"