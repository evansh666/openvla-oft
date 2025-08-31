## Clone OpenVLA-OFT and RLDS_DATASET_BUILDER
```
git clone https://github.com/moojink/openvla-oft
git clone https://github.com/moojink/rlds_dataset_builder
```

## Environment Setup
```
conda deactivate
conda create -n openvla-oft python=3.10 -y
conda activate openvla-oft

pip3 install torch torchvision torchaudio

cd openvla-oft
pip install -e .

pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation

# install for RLDS dataset: tensorflow, tensorflow_datasets, tensorflow_hub, apache_beam
pip install tensorflow_hub
pip install apache_beam

```

## Convert data from hdf5 to RLDS 
1. See instructions for converting to RLDS [here](RLDS_builder/README.md). 

2. A sample BEHAVIOR data to RLDS conversion script is available [here](RLDS_builder/behavior_dataset/behavior_turn_on_radio/), you can use the following code to get RLDS-formatted data:

```
cd RLDS_builder
tfds build --data_dir /path/to/save/rlds/dataset
```

2. If you want to customize your own dataset, revise the dataset builder (e.g., ['behavior_turn_on_radio_dataset_builder.py'](RLDS_builder/behavior_dataset/behavior_turn_on_radio/behavior_turn_on_radio_dataset_builder.py). 



## Finetune OpenVLA with OFT+
1. Register the dataset (e.g. behavior_turn_on_radio) with openvla-oft dataloader by adding an entry in the following files:
    - Add an entry in StateEncoding and ActionEncoding; and Add a data name mapping in `configs.py` ([here](prismatic/vla/datasets/rlds/oxe/configs.py#L711))
    - Add data transform in `transforms.py` ([here](prismatic/vla/datasets/rlds/oxe/transforms.py#L937)) 
    - Add data mixture proportion in `mixtures.py` ([here](prismatic/vla/datasets/rlds/oxe/mixtures.py#L231)).
    - Set constants of BEHAVIOR, e.g., desired action chunk size ([here]([`prismatic/vla/constants.py`]))
    - Add normalize and absolute action mask in `materialize.py` ([here](prismatic/vla/datasets/rlds/oxe/materialize.py)).
    - Add behavior in three camera views selection ([here](prismatic/vla/datasets/datasets.py#L116))

3. Revise dataset and setting in [finetune.sh](finetune.sh). For more detailed parameter selection, please refer [OpenVLA-Finetune Instruction](https://github.com/moojink/openvla-oft/blob/main/ALOHA.md).
```
./finetune.sh
```


## Evaluation in Omnigibson
1. Deploy finetuned checkpoint:
```
python vla-scripts/deploy.py \
  --pretrained_checkpoint /PATH/TO/FINETUNED/MODEL/CHECKPOINT/DIR/ \
  --use_l1_regression True \
  --use_film True \
  --num_images_in_input 3 \
  --use_proprio True \
  --center_crop True \
  --unnorm_key behavior_turn_on_radio
  ```
  This opens a connection listening on 0.0.0.0:8777.

2. Run the evaluation  

