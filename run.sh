#!/bin/sh

# if [[ $# -eq 0 ]]; then
#   echo "Usage: $0 <task_to_run> [args...]"
#   exit 1
# fi

# task_to_run=$1
# shift

# if [ -f ".env" ]; then
# 	set -a
# 	source .env
#     echo "Loaded environment variables from .env"
# 	set +a
# else
# 	echo ".env file not found in $(pwd). Create one with HF_API_TOKEN and other vars." >&2
# 	exit 1
# fi

# if [ -z "${HF_API_TOKEN:-}" ]; then
# 	echo "HF_API_TOKEN is not set. Please populate .env with your Hugging Face token." >&2
# 	exit 1
# fi

# python3 task0.py \
#     --hf-token "$HF_API_TOKEN" \
#     --method greedy \
#     --A 20 \
#     --B 8 \
#     --tau 0.5 \
#     --out data/outputs_task0_greedy.jsonl

# python3 eval.py --inputs data/outputs_task0_greedy.jsonl --counts-dir tinystories_ngrams --model meta-llama/Meta-Llama-3-8B-Instruct --hf-token $HF_API_TOKEN --out data/metrics_greedy.json

# python3 task0.py \
#     --hf-token "$HF_API_TOKEN" \
#     --method temperature \
#     --A 20 \
#     --B 8 \
#     --tau 1.0 \
#     --out data/outputs_task0_temperature_1.0.jsonl

# python3 eval.py --inputs data/outputs_task0_temperature_1.0.jsonl --counts-dir tinystories_ngrams --model meta-llama/Meta-Llama-3-8B-Instruct --hf-token $HF_API_TOKEN --out data/metrics_temp_1.0.json

# python3 task0.py \
#     --hf-token "$HF_API_TOKEN" \
#     --method temperature \
#     --A 20 \
#     --B 8 \
#     --tau 0.5 \
#     --out data/outputs_task0_temperature_0.5.jsonl

# python3 eval.py --inputs data/outputs_task0_temperature_0.5.jsonl --counts-dir tinystories_ngrams --model meta-llama/Meta-Llama-3-8B-Instruct --hf-token $HF_API_TOKEN --out data/metrics_temp_0.5.json

# python3 task0.py \
#     --hf-token "$HF_API_TOKEN" \
#     --method temperature \
#     --A 20 \
#     --B 8 \
#     --tau 0.9 \
#     --out data/outputs_task0_temperature_0.9.jsonl

# python3 eval.py --inputs data/outputs_task0_temperature_0.9.jsonl --counts-dir tinystories_ngrams --model meta-llama/Meta-Llama-3-8B-Instruct --hf-token $HF_API_TOKEN --out data/metrics_temp_0.9.json

# python3 task0.py \
#     --hf-token "$HF_API_TOKEN" \
#     --method topk \
#     --A 20 \
#     --B 8 \
#     --k 5 \
#     --out data/outputs_task0_topk_5.jsonl

# python3 eval.py --inputs data/outputs_task0_topk_5.jsonl --counts-dir tinystories_ngrams --model meta-llama/Meta-Llama-3-8B-Instruct --hf-token $HF_API_TOKEN --out data/metrics_topk_5.json

# python3 task0.py \
#     --hf-token "$HF_API_TOKEN" \
#     --method topk \
#     --A 20 \
#     --B 8 \
#     --k 10 \
#     --out data/outputs_task0_topk_10.jsonl

# python3 eval.py --inputs data/outputs_task0_topk_10.jsonl --counts-dir tinystories_ngrams --model meta-llama/Meta-Llama-3-8B-Instruct --hf-token $HF_API_TOKEN --out data/metrics_topk_10.json

# Hide the token value from process list
# HF_API_TOKEN=${HF_API_TOKEN:-}

# python task1.py \
#     --counts-dir tinystories_ngrams \
#     --A 20 \
#     --B 8 \
#     --beta 5.0 \
#     --k 10 \
#     --out data/outputs_task1_IS_5.jsonl

# python3 eval.py \
#     --inputs data/outputs_task3_TSMC.jsonl \
#     --counts-dir tinystories_ngrams \
#     --model meta-llama/Meta-Llama-3-8B-Instruct \
#     --out data/metrics_task3_TSMC.json

python task3.py \
    --counts-dir tinystories_ngrams \
    --A 1 \
    --B 8 \
    --beta 5.0 \
    --k 10 \
    --out data/outputs_task3_TSMC.jsonl