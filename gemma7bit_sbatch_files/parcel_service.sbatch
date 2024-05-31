#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=47:00:00
#SBATCH --job-name=gemma7bitparcel_service


cd ../llm-bias
python3.8 short_completions gemma7bit parcel_service
