## BEHAVIOR-1K Challenge -- Finetune OpenVLA-OFT
This repo provides a simplest version instruction to finetune OpenVLA-OFT on BEHAVIOR-1K Challenge dataset. 

## Repo Clone
```
git clone https://github.com/evansh666/openvla-oft.git
git clone https://github.com/StanfordVL/BEHAVIOR-1K.git
```
You can also start with the original openvla-oft [repo](https://github.com/moojink/openvla-oft). This finetuning instruction is adapted from the [ALOHA finetuning task](https://github.com/moojink/openvla-oft/blob/main/ALOHA.md).


## Installation
```
conda deactivate
conda create -n openvla-oft python=3.10 -y
conda activate openvla-oft

cd openvla-oft
pip install -e .

pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation

# Install for RLDS dataset: tensorflow, tensorflow_datasets, tensorflow_hub, apache_beam
pip install tensorflow_hub
pip install apache_beam

# Install Behavior env for server deploy 
cd ../BEHAVIOR-1K # go to BEHAVIOR-1K directory

# Install bddl
cd bddl
pip install -e .
pip install pymeshlab==2022.2.post4

# Install omnigibson with eval dependencies
cd Omnigibson
pip install .
```

## Data Conversion
Since Openvla-oft requires RLDS dataset, we need first convert BEHAVIOR dataset into RLDS format. 

1. See instructions for converting to RLDS [here](RLDS_builder/README.md). 

2. A sample BEHAVIOR data to RLDS conversion script is available [here](RLDS_builder/behavior_dataset/behavior_turn_on_radio/), you can use the following code to get RLDS-formatted data:

```
cd RLDS_builder
cd behavior_dataset/behavior_turn_on_radio
tfds build --data_dir /path/to/save/rlds/dataset
```

3. If you want to customize your own dataset, revise the dataset builder (e.g., ['behavior_turn_on_radio_dataset_builder.py'](RLDS_builder/behavior_dataset/behavior_turn_on_radio/behavior_turn_on_radio_dataset_builder.py). 



## Finetune OpenVLA-OFT+
There are a few files in OpenVLA-OFT+ needs change to adapt our new robot:

1. Register the dataset (e.g. behavior_turn_on_radio) with openvla-oft dataloader by adding an entry in the following files:
    - Add an entry in StateEncoding and ActionEncoding; and Add a data name mapping in `configs.py` ([here](prismatic/vla/datasets/rlds/oxe/configs.py#L711))
    - Add data transform in `transforms.py` ([here](prismatic/vla/datasets/rlds/oxe/transforms.py#L937)) 
    - Add data mixture proportion in `mixtures.py` ([here](prismatic/vla/datasets/rlds/oxe/mixtures.py#L231)).
    - Set constants of BEHAVIOR, e.g., desired action chunk size ([here]([`prismatic/vla/constants.py`]))
    - Add normalize and absolute action mask in `materialize.py` ([here](prismatic/vla/datasets/rlds/oxe/materialize.py)).
    - Add behavior in three camera views selection ([here](prismatic/vla/datasets/datasets.py#L116))

3. Revise dataset and setting in [finetune.sh](finetune.sh). For more detailed parameter selection, please refer [OpenVLA-Finetune Instruction](https://github.com/moojink/openvla-oft/blob/main/ALOHA.md).

```
torchrun --standalone --nnodes 1 --nproc-per-node X vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /PATH/TO/RLDS/DATASETS/DIR/ \
  --dataset_name /YOUR/DATASET/NAME \
  --run_root_dir /YOUR/CHECKPOINTS/AND/LOG/DIR/ \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film True \
  --num_images_in_input 3 \
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
  --wandb_entity "YOUR_WANDB_ENTITY" \
  --wandb_project "YOUR_WANDB_PROJECT" \
  --run_id_note parallel_dec--25_acts_chunk--continuous_acts--L1_regression--3rd_person_img--left_right_wrist_imgs--proprio_state--film
```


## Evaluation
After finetuning, you can deploy the 
1. Deploy finetuned checkpoint:
```
python vla-scripts/deploy.py \
  --pretrained_checkpoint /PATH/TO/FINETUNED/MODEL/CHECKPOINT/DIR/ \
  --use_l1_regression True \
  --use_film True \
  --num_images_in_input 3 \
  --use_proprio True \
  --center_crop True \
  --unnorm_key /NAME/OF/DATASET

  # Or directly run after modifying deploy.sh
  ./scripts/deploy.sh
  ```
  This opens a connection listening on 0.0.0.0:8000.


2. Run the evaluation on BEHAVIOR
```
# activate your installed behavior environment
# if you haven't install behavior environment, check https://github.com/StanfordVL/BEHAVIOR-1K
conda deactivate
conda activate behavior 

# run eval script
cd BEHAVIOR-1K/Omnigibson/omnigibson/learning
python eval.py policy=websocket task.name=turning_on_radio
```
