#!/bin/bash

#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16$
#SBATCH --mem=32000M        # memory per node
#SBATCH --time=0-01:00      # time (DD-HH:MM)
#SBATCH --output=output.out  # %N for node name, %j for jobID

#### local path
#RACE_DIR=../RACE
#INIT_CKPT_DIR=../xlnet_cased_L-24_H-1024_A-16

#### google storage path
GS_ROOT=${STORAGE_BUCKET}
GS_INIT_CKPT_DIR=${GS_ROOT}/xlnet_cased_L-24_H-1024_A-16
GS_PROC_DATA_DIR=${GS_ROOT}/proc_data/race
GS_MODEL_DIR=${GS_ROOT}/experiment/race

# TPU name in google cloud
TPU_NAME=

python3 estimator.py \
  --data_dir=data/ \
  --model_dir=model/ \
  --train_steps=100000 \
  --dropout=0.5 \
  --heads=4 \
  --seq_len=40 \
  --batch_size=128 \
  --layers=2 \
  --depth=16 \
  --feedforward=16 \
  --train=True \
  --evaluate=True \
  --predict=True \

  $@
