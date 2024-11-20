#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00
#SBATCH --job-name=justice_mistral


cd ../llm-bias
python3.8 short_completions_other.py mistral justice
