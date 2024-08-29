#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=47:00:00
#SBATCH --job-name=otherllama3itcolonialism


cd ../llm-bias
python3.8 short_completions_other.py llama3it colonialism
