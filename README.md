## Environment Setup
```
./setup_env.sh
```

## Convert data from hdf5 to RLDS 
1. Check [README.md](RLDS_builder/README.md)

2. Revise dataset builder (e.g., ['b1k_dataset_dataset_builder.py'](RLDS_builder/b1k_dataset/b1k_dataset_dataset_builder.py)

```
cd RLDS_builder
# do any customization to the dataset
tfds build --data_dir /path/to/save/rlds/dataset
```
To apply to our customized settings for finetuning, the dataset needs start with "b1k"

## Finetune OpenVLA with OFT+
1. Register the dataset (e.g. b1k_pick_up_trash_100_demos) with our dataloader by adding an entry in `configs.py` ([here](prismatic/vla/datasets/rlds/oxe/configs.py#L680)), `transforms.py` ([here](prismatic/vla/datasets/rlds/oxe/transforms.py#L928)), and `mixtures.py` ([here](prismatic/vla/datasets/rlds/oxe/mixtures.py#L216)).

2. Set desired action chunk size in [`prismatic/vla/constants.py`](prismatic/vla/constants.py)

(Please check this ['commit'](https://github.com/evansh666/openvla-oft/commit/a2756d61a4dc7bcfe617e28a42d0f5c6c32a5997#diff-b40d80fa2ae97b52299de25975b4fe19f9cd789f70d5719e1ecfbf828c99068b) for above revision reference. 

3. Revise dataset and setting in [finetune.sh](finetune.sh)
```
./finetune.sh
```
