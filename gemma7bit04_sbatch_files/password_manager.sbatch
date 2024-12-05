#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00
#SBATCH --job-name=password_manager_gemma7bit04


cd ../llm-bias
python3.8 short_completions.py gemma7bit04 password_manager
