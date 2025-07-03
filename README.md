## Environment Setup
```
./setup_env.sh
```

## Convert data from hdf5 to RLDS 
check [README.md](RLDS_builder/README.md)
```
cd RLDS_builder
# do any customization to the dataset
tfds build --data_dir /path/to/save/rlds/dataset
```

## Finetune OpenVLA with OFT+
revise dataset and setting in [finetune.sh](finetune.sh)
```
./finetune.sh
```