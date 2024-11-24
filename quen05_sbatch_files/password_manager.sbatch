#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00
#SBATCH --job-name=password_manager_quen05


cd ../llm-bias
python3.8 short_completions.py quen05 password_manager
