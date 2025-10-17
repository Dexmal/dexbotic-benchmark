#!/bin/bash

conda init && source activate
conda activate RoboTwin

policy_name=dexbotic
config=${1}
ckpt_setting=dexbotic
gpu_id=${2:-0}

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

python evaluation/run_robotwin2_evaluation.py --config ${config} \
    --set ckpt_setting ${ckpt_setting} \
    --set policy_name ${policy_name} 