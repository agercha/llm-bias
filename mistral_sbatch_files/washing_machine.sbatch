#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00
#SBATCH --job-name=washing_machine_mistral


cd ../llm-bias
python3.8 short_completions.py mistral washing_machine
