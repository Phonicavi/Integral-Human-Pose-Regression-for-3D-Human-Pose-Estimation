#!/bin/sh

srun --partition=Pose --gres=gpu:8 --job-name=Integral python main.py --gpu 0-7 --continue
#srun --partition=Pose --gres=gpu:6 --job-name=Integral python main.py --gpu 0-5 --baseline
