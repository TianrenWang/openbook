#!/bin/bash

#SBATCH --gres=gpu:p100:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=3   # maximum CPU cores per GPU request: 6 on Cedar, 16$
#SBATCH --mem=12000M        # memory per node
#SBATCH --time=0-08:00      # time (DD-HH:MM)
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

python3 kmeans_estimator.py \
  --data_dir=data/ \
  --graph_dir=knowledge_graph/ \
  --embed_steps=100 \
  --graph_size=512 \
  --embed=True \
  --predict=False \
  --predict_samples=10 \
  --description="Put experiment description here" \

  $@
