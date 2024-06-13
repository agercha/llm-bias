#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00
#SBATCH --job-name=llama3itnew_air_fryer_short


cd ../llm-bias
python3.8 short_completions.py llama3itnew hiking_shoes
