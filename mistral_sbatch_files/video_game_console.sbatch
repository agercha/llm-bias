#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00
#SBATCH --job-name=video_game_console_mistral


cd ../llm-bias
python3.8 short_completions.py mistral video_game_console
