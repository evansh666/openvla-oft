import h5py
import numpy as np
import pandas as pd
import re
from collections import OrderedDict
from RLDS_builder.behavior_dataset.utils.conversion_utils import resize_with_pad
from RLDS_builder.behavior_dataset.utils.obs_utils import RGBVideoLoader, DepthVideoLoader

IMAGE_SIZE = 256
ROBOT_CAMERA_NAMES = {
    "left_wrist": "robot_r1::robot_r1:left_realsense_link:Camera:0",
    "right_wrist": "robot_r1::robot_r1:right_realsense_link:Camera:0",
    "head": "robot_r1::robot_r1:zed_link:Camera:0",
}
HEAD_RESOLUTION = (720, 720)
WRIST_RESOLUTION = (480, 480)
# Action indices
ACTION_QPOS_INDICES = {
    "R1Pro": OrderedDict(
        {
            "base": np.s_[0:3],
            "torso": np.s_[3:7],
            "left_arm": np.s_[7:14],
            "left_gripper": np.s_[14:15],
            "right_arm": np.s_[15:22],
            "right_gripper": np.s_[22:23],
        }
    )
}

# Proprioception indices
PROPRIO_QPOS_INDICES = {
    "R1Pro": OrderedDict(
        {
            "torso": np.s_[6:10],
            "left_arm": np.s_[10:24:2],
            "right_arm": np.s_[11:24:2],
            "left_gripper": np.s_[24:26],
            "right_gripper": np.s_[26:28],
        }
    )
}

def load_video(data_path, task_id, demo_id, camera_names):
    images = {}
    for camera_id, camera_name in camera_names.items():
        try:
            rgb_video_loader = RGBVideoLoader(
                data_path=data_path,
                task_id=task_id,
                camera_id=camera_id,
                demo_id=demo_id,
                # batch_size=8,         # Optional: None loads all at once
                # stride=1,             # Optional: use stride for overlapping batches
                output_size=(224, 224) 
            )
            images[camera_id] = rgb_video_loader.frames
            images[camera_id] = images[camera_id].permute(0, 2, 3, 1)
        except Exception as e:
            print(f"Error loading video for camera {camera_id}: {e}")
            return None
    return images

def load_low_dim_from_parquet(data_folder, task_id, demo_id):
    try:
        df = pd.read_parquet(f"{data_folder}/data/task-{task_id:04d}/episode_{task_id:04d}{demo_id:04d}.parquet")
        T = len(df["index"])
        actions = df["action"].to_numpy()
        proprio = df["observation.state"].to_numpy()
        cam_rel_poses = df["observation.cam_rel_poses"].to_numpy()
        task_info = df["observation.task_info"].to_numpy()
    except Exception as e:
        print(f"Error loading parquet file: {e}")
        return None
    
    return (actions, proprio)

def process_proprio_state(proprio):
    base_qpos = proprio[:,246:249] # 3
    trunk_qpos = proprio[:,238:242] # 4
    arm_left_qpos = proprio[:,158:165] #  7
    arm_right_qpos = proprio[:,198:205] #  7
    left_gripper_width = proprio[:,194:196].sum(axis=-1)[:,None] # 1
    right_gripper_width = proprio[:,234:236].sum(axis=-1)[:,None] # 1
    
    prop_state = np.concatenate((base_qpos, trunk_qpos, arm_left_qpos, arm_right_qpos, left_gripper_width, right_gripper_width), axis=-1) # 23
    return prop_state

def create_episode_from_video(video_path, language_instruction):
    assert video_path is not None, "Video path is required"
    assert language_instruction is not None, "Language instruction is required"
    
    camera_ids = list(ROBOT_CAMERA_NAMES.keys())
    m = re.search(r"^(.*)/videos/task-(\d{4})/observation\.images\.rgb\.(.*?)/episode_(\d{4})(\d{4})\.mp4$", video_path)
    if not m:
        print(f"Invalid video path: {video_path}")
        return None
    
    data_path, task_id, _, _, demo_id = m.groups()
    task_id = int(task_id)
    demo_id = int(demo_id)
    images = load_video(data_path, task_id, demo_id, ROBOT_CAMERA_NAMES)
    actions, state = load_low_dim_from_parquet(data_path, task_id, demo_id)
    state = process_proprio_state(np.stack(state))
    if images is None or state is None:
        return None
    
    num_steps = actions.shape[0]
    episode = []
    for i in range(num_steps):
        episode.append({
            'observation': {
                **{'state': state[i].astype(np.float32)},
                **{k: resize_with_pad(images[k][i].numpy(), IMAGE_SIZE, IMAGE_SIZE) for k in camera_ids if k in images}
            },
            'action': np.asarray(actions[i], dtype=np.float32),
            'language_instruction': language_instruction,
            'discount': 1.0,
            'reward': 1.0,
            'is_first': i == 0,
            'is_last': i == (num_steps - 1),
            'is_terminal': i == (num_steps - 1),
        })
        
    return episode

