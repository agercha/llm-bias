#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00
#SBATCH --job-name=32_3llama_short


cd ../llm-bias
python3.8 short_completions.py llama3 sparkling_water
