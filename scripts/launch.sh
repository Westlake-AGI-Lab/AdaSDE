#!/usr/bin/env bash
set -e

# Minimal knobs you may want to tweak
DATASET_NAME=${DATASET_NAME:-cifar10}
BATCH=${BATCH:-32}
TOTAL_KIMG=${TOTAL_KIMG:-10}

# Training hyperparameters (match your example)
SOLVER_FLAGS="--sampler_stu=adasde --sampler_tea=dpm --num_steps=3 --M=3 --afs=True --scale_dir=0.05 --scale_time=0.5 --gamma=0.02 --seed=0 --lr=0.2 --coslr"
SCHEDULE_FLAGS="--schedule_type=time_uniform --schedule_rho=1"

# Single node, single GPU; change port if needed
torchrun --standalone --nproc_per_node=2 --master_port=11111 \
  train.py --dataset_name="$DATASET_NAME" --batch="$BATCH" --total_kimg="$TOTAL_KIMG" \
  $SOLVER_FLAGS $SCHEDULE_FLAGS "$@"
