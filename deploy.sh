#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate openvla-oft

DATASET_NAME=behavior_turn_on_radio
CHECKPOINT_NAME=openvla-7b+behavior_turn_on_radio+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--10_acts_chunk--continuous_acts--l1--3rd_person_img--left_right_wrist_imgs--proprio_state--film--35000_chkpt/
CHECKPOINT_DIR=checkpoints/${CHECKPOINT_NAME}
python vla-scripts/deploy.py \
  --pretrained_checkpoint ${CHECKPOINT_DIR} \
  --use_l1_regression True \
  --use_film True \
  --num_images_in_input 3 \
  --use_proprio True \
  --center_crop True \
  --unnorm_key ${DATASET_NAME}
~
