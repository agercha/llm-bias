#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00
#SBATCH --job-name=rental_car_gemma7bit10


cd ../llm-bias
python3.8 short_completions.py gemma7bit10 rental_car
