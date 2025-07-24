import h5py
import json
import numpy as np


def create_episode_from_hdf5(
        input_path, 
        episode_id,
        data_grp_key="data",
        obs_key="obs",
        image_keys=["image", "left_wrist_image", "right_wrist_image", "low_cam_image"],
        action_key="action",
        state_key="state",
        state_size_key="state_size",
        reward_key="reward",
        terminated_key="terminated",
        language_instruction="",
    ):
    input_hdf5 = h5py.File(input_path, "r")
    data_grp = input_hdf5[data_grp_key]
    assert f"demo_{episode_id}" in data_grp, f"No valid episode with ID {episode_id} found!"
    traj_grp = data_grp[f"demo_{episode_id}"]

    # Grab episode data
    # Skip early if found malformed data
    try:
        assert 'action' in traj_grp, "action key not found in data"
        assert 'state' in traj_grp, "state key not found in data"
        
        observation = traj_grp[obs_key][()]
        action = traj_grp[action_key][()]
        state = traj_grp[state_key][()]
        state_size = traj_grp[state_size_key][()]
        reward = traj_grp[reward_key][()]
        terminated = traj_grp[terminated_key][()]

        # Build episode steps
        num_steps = action.shape[0]
        episode = []
        
        for i in range(num_steps):
            episode.append({
                'observation': {
                    **{'state': state[i][:state_size].astype(np.float32)},
                    **{k: observation[k][i] for k in image_keys if k in observation}
                },
                'action': np.asarray(action[i], dtype=np.float32),
                'language_instruction': language_instruction,
                'discount': 1.0,
                'reward': reward[i],
                'is_first': i == 0,
                'is_last': i == (num_steps - 1),
                'is_terminal': terminated[i],
            })
            
    except KeyError as e:
        print(f"Got error when trying to load episode {episode_id}:")
        print(f"Error: {str(e)}")
        return None
    
    return episode