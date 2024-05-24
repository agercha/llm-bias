#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00
#SBATCH --job-name=email_providers_3itllama_short


cd ../llm-bias
python3.8 short_completions.py llama3it email_providers
