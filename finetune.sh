eval "$(conda shell.bash hook)"
conda activate openvla-oft

NUM_GPUS=4
DATASET_ROOT_PATH=/home/evans/projects/openvla-oft/datasets
DATASET_NAME=b1k_pick_up_trash_100_demos
CHECKPOINT_PATH=/home/evans/projects/openvla-oft/checkpoints
WANDB_ENTITY=evansh666
WANDB_PROJECT=OpenVLA-OFT
RUN_ID=parallel_dec--10_acts_chunk--continuous_acts--L1_regression--3rd_person_img--left_right_wrist_imgs--proprio_state--film

INPUT_NUM_IMGS=3

torchrun --standalone --nnodes 1 --nproc-per-node $NUM_GPUS vla-scripts/finetune.py \
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
  --use_val_set True \
  --val_freq 10000 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity $WANDB_ENTITY \
  --wandb_project $WANDB_PROJECT \
  --run_id_note $RUN_ID 
