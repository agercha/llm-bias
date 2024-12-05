#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00
#SBATCH --job-name=chip_gemma7bit09


cd ../llm-bias
python3.8 short_completions.py gemma7bit09 chip
