#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00
#SBATCH --job-name=credit_card_mistral


cd ../llm-bias
python3.8 short_completions.py mistral credit_card
