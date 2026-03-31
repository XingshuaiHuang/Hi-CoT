#!/bin/bash
export PYTHONPATH=.:$PYTHONPATH

for model in Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-4B Qwen/Qwen3-8B Qwen/Qwen3-14B Qwen/Qwen3-4B-Thinking-2507 Qwen/Qwen3-4B-Instruct-2507 Qwen/Qwen3-32B deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B deepseek-ai/DeepSeek-R1-Distill-Qwen-7B deepseek-ai/DeepSeek-R1-Distill-Llama-8B deepseek-ai/DeepSeek-R1-Distill-Qwen-14B deepseek-ai/DeepSeek-R1-Distill-Qwen-32B; do
    for template in cot ps hicot_wo_structure hicot; do
        echo
        echo ------------- model: $model, template: $template -------------
        echo
        CUDA_VISIBLE_DEVICES=7 python scripts/evaluate_model.py --model_name $model --template $template --save True
    done
done