#!/bin/sh
srun --partition=Pose --gres=gpu:8 --job-name=Integral python main.py --gpu 0-7 --continue
