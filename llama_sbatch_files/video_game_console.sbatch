'#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=47:00:00
#SBATCH --job-name=llamavideo_game_console


cd ../llm-bias
python3.8 short_completions llama video_game_console
