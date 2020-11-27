#!/bin/bash
conda activate /data/saandeepaath/my_envs/.dist
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 /data/saandeepaath/cbasics/main.py
conda deactivate