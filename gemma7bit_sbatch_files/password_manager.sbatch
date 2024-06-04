#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00
#SBATCH --job-name=air_fryer_gemma7bit_short


cd ../llm-bias
python3.8 short_completions.py gemma7bit air_fryer
