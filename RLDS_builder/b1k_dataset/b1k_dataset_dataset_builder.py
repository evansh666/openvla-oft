from typing import Iterator, Tuple, Any

import h5py
import glob
import numpy as np
import tensorflow_datasets as tfds

from RLDS_builder.b1k_dataset.utils.conversion_utils import MultiThreadedDatasetBuilder, resize_with_pad
from data_utils import create_episode_from_hdf5

# Change following parameters to match your dataset
IMAGE_SIZE = 256
NUM_ACTIONS_CHUNK = 10
ACTION_DIM = 23
STATE_DIM = 23

TRAIN_DATA_PATH = "./put_green_pepper_into_pot/train/*.hdf5"
VAL_DATA_PATH = None

# Data conversion configuration
# key: rlds key, value: hdf5 key
KEY_MAP = {
    'images': {
        'image': "observation/egocentric_camera",
        'left_wrist_image': "observation/wrist_image_left",
        'right_wrist_image': "observation/wrist_image_right",
        'low_cam_image': None,
    },
    'array': {
        'action': "actions",   
        'state': "state",
        'reward': "reward",
        'discount': None,
    },
    'scalar': {
        'language_instruction': "prompt",
    }
}



def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Yields episodes for list of data paths."""

    def _parse_example(episode_path):
        # Load and process all data
        with h5py.File(episode_path, "r") as f:
            episode = create_episode_from_hdf5(f)
        
        sample = {
            'steps': episode,
            'episode_metadata': {
                'file_path': episode_path
            }
        }

        # If you want to skip an example for whatever reason, simply return None
        return episode_path, sample

    for path in paths:
        ret = _parse_example(path)
        yield ret


class B1KDataset(MultiThreadedDatasetBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    N_WORKERS = 40             # number of parallel workers for data conversion
    MAX_PATHS_IN_MEMORY = 80   # number of paths converted & stored in memory before writing to disk
                               # -> the higher the faster / more parallel conversion, adjust based on avilable RAM
                               # note that one path may yield multiple episodes and adjust accordingly
    PARSE_FCN = _generate_examples      # handle to parse function from file paths to RLDS episodes

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        **{obs_key: tfds.features.Image(
                            shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc=f'{obs_key.replace("_", " ").title()} RGB observation.',
                        ) for obs_key, hdf5_key in KEY_MAP['images'].items() if hdf5_key is not None},
                        'state': tfds.features.Tensor(
                            shape=(STATE_DIM,),
                            dtype=np.float32,
                            doc='Robot joint state (7D left arm + 7D right arm).',
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(ACTION_DIM,),
                        dtype=np.float32,
                        doc='Robot arm action.',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    
    def _split_paths(self):
        """Define filepaths for data splits."""
        paths = {}
        
        if TRAIN_DATA_PATH:
            paths["train"] = glob.glob(TRAIN_DATA_PATH)
        if VAL_DATA_PATH:
            paths["val"] = glob.glob(VAL_DATA_PATH)
            
        return paths
            