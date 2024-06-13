#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00
#SBATCH --job-name=llama_air_fryer_short


cd ../llm-bias
python3.8 short_completions.py llama gas_station
