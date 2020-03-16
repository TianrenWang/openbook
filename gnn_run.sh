#!/bin/bash

#SBATCH --gres=gpu:v100:4        # request GPU "generic resource"
#SBATCH --cpus-per-task=12   # maximum CPU cores per GPU request: 6 on Cedar, 16$
#SBATCH --mem=50000M        # memory per node
#SBATCH --time=0-4:00      # time (DD-HH:MM)
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

python3 gnn_estimator.py \
  --data_dir=data/ \
  --graph_dir=knowledge_graph/ \
  --model_dir=gnn_model/ \
  --train_steps=50000 \
  --dropout=0.5 \
  --seq_len=80 \
  --recurrences=5 \
  --batch_size=32 \
  --learning_rate=1e-5 \
  --train=True \
  --predict=True \
  --predict_samples=10 \
  --description="Put experiment description here" \

  $@
