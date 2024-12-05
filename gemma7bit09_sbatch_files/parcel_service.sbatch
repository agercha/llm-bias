#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00
#SBATCH --job-name=parcel_service_gemma7bit09


cd ../llm-bias
python3.8 short_completions.py gemma7bit09 parcel_service
