#!/bin/bash

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
  --data-dir=data/ \
  --model-dir=model/ \
  --train-steps=4000 \
  $@
