#!/bin/bash

#SBATCH --gres=gpu:p100:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16$
#SBATCH --mem=32000M        # memory per node
#SBATCH --time=0-04:00      # time (DD-HH:MM)
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
  --train_steps=200000 \
  --vocab_level=13 \
  --dropout=0.1 \
  --heads=8 \
  --seq_len=40 \
  --sparse_len=8 \
  --sparse_lim=6 \
  --use_sparse=False \
  --sparse_thresh=0.0 \
  --conc=1.4 \
  --sparse_loss=1 \
  --batch_size=128 \
  --layers=4 \
  --depth=128 \
  --feedforward=512 \
  --position_encode=1 \
  --train=True \
  --predict=True \
  --predict_samples=10 \

  $@
