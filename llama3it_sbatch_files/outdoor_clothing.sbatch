#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00
#SBATCH --job-name=outdoor_clothing_3itllama_short


cd ../llm-bias
python3.8 short_completions.py llama3it outdoor_clothing
