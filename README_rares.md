export HF_HOME="/home/rares/LLaMA-Factory/large_data/cache_hugg"

Example evaluation:
```
https://github.com/knoveleng/open-rs/blob/main/src/open_r1/evaluate.py
```
## Evaluation
```
bash ./run_sh/gsmk8.sh
```

# Training
1. Finetuning
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml

2. MATH PPO
CUDA_VISIBLE_DEVICES="7" llamafactory-cli train examples/var_exp/qwen_demo_ppo.yaml