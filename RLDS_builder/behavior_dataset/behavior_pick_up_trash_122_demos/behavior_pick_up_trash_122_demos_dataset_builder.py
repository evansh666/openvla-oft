from typing import Iterator, Tuple, Any
import glob
import numpy as np
import tensorflow_datasets as tfds

from RLDS_builder.behavior_dataset.utils.conversion_utils import MultiThreadedDatasetBuilder, resize_with_pad
from RLDS_builder.behavior_dataset.utils.data_utils import create_episode_from_hdf5, create_episode_from_video, ROBOT_CAMERA_NAMES

# Change following parameters to match your dataset
IMAGE_SIZE = 256
NUM_ACTIONS_CHUNK = 10
ACTION_DIM = 23
STATE_DIM = 23

TRAIN_DATA_PATH = "/vision/u/wsai/data/behavior/videos/task-0001/observation.images.rgb.head"
VAL_DATA_PATH = ""
SOURCE = "video"
INSTRUCTION = "pick up trash"
        
def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Yields episodes for list of data paths."""
    def _parse_example(episode_path, source=SOURCE):
        if source == "video":
            episode = create_episode_from_video(episode_path)
            if episode is None:
                return None
            
            sample = {
            'steps': episode,
            'episode_metadata': {
                'file_path': episode_path,
            }}
        elif source == "parquet":
            episode = create_episode_from_hdf5(episode_path)
            sample = {
                'steps': episode,
                'episode_metadata': {
                    "file_path": episode_path
                }
            }
        return episode_path, sample # Return None to skip an example

    for path in paths:
        ret = _parse_example(path)
        yield ret


class behavior_pick_up_trash_122_demos(MultiThreadedDatasetBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    N_WORKERS = 1             # number of parallel workers for data conversion
    MAX_PATHS_IN_MEMORY = 1   # number of paths converted & stored in memory before writing to disk
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
                        ) for obs_key in list(ROBOT_CAMERA_NAMES.keys())},
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
  
        if SOURCE == "video":
            suffix = "mp4"
        elif SOURCE == "hdf5":
            suffix = "hdf5"
        else:
            raise ValueError(f"Invalid source: {SOURCE}")

        if TRAIN_DATA_PATH:
            paths["train"] = glob.glob(f"{TRAIN_DATA_PATH}/*.{suffix}")
        if VAL_DATA_PATH:
            paths["val"] = glob.glob(f"{VAL_DATA_PATH}/*.{suffix}")
        
        return paths
            
