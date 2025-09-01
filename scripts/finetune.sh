#!/bin/bash

#SBATCH --job-name=ft-openvla
#SBATCH --account=viscam
#SBATCH --partition=viscam
#SBATCH --exclude=svl13,svl12
#SBATCH --nodes=1
#SBATCH --gres=gpu:titanrtx:4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=350G
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm_logs/finetune.out
#SBATCH --error=slurm_logs/finetune.err

eval "$(conda shell.bash hook)"
conda activate openvla-oft

DATASET_ROOT_PATH=/vision/u/yinhang/data/openvla
DATASET_NAME=behavior_turn_on_radio
CHECKPOINT_PATH=/vision/u/yinhang/openvla-oft/checkpoints
WANDB_ENTITY=evansh666-stanford-university
WANDB_PROJECT=OpenVLA-OFT
RUN_ID=10_acts_chunk--continuous_acts--L1_regression--3img--proprio_state--film
INPUT_NUM_IMGS=3
mkdir -p $CHECKPOINT_PATH

export HF_HOME=/vision/u/yinhang/cache/huggingface

torchrun --standalone --nnodes 1 --nproc-per-node $SLURM_GPUS_ON_NODE vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir $DATASET_ROOT_PATH \
  --dataset_name $DATASET_NAME \
  --run_root_dir $CHECKPOINT_PATH \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film True \
  --num_images_in_input $INPUT_NUM_IMGS \
  --use_proprio True \
  --batch_size 4 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 50000 \
  --max_steps 100005 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --lora_rank 32 \
  --run_id_note $RUN_ID \
  --wandb_entity $WANDB_ENTITY \
  --wandb_project $WANDB_PROJECT \
  --image_aug True 
  # --use_val_set True \
  # --val_freq 10000 \
