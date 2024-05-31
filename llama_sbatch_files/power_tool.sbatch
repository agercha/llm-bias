#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=47:00:00
#SBATCH --job-name=llamapower_tool


cd ../llm-bias
python3.8 short_completions llama power_tool
