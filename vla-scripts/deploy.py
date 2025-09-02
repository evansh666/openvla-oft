"""
Starts VLA server which the client can query to get robot actions.
"""

import logging
import numpy as np
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union

import draccus
import torch
from experiments.robot.openvla_utils import (
    get_vla,
    get_vla_action,
    get_action_head,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    get_image_resize_size,
)
from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX

import sys
sys.path.insert(0, "/vision/u/yinhang/BEHAVIOR-1K/OmniGibson")
from omnigibson.learning.utils.network_utils import WebsocketPolicyServer
from omnigibson.learning.utils.eval_utils import PROPRIOCEPTION_INDICES, ROBOT_CAMERA_NAMES


# === Policy Wrapper for WebSocket Server ===
class OpenVLAPolicy:
    def __init__(self, cfg) -> None:
        self.cfg = cfg

        # Load model
        self.vla = get_vla(cfg)

        # Load proprio projector
        self.proprio_projector = None
        if cfg.use_proprio:
            self.proprio_projector = get_proprio_projector(cfg, self.vla.llm_dim, PROPRIO_DIM)

        # Load continuous action head
        self.action_head = None
        if cfg.use_l1_regression or cfg.use_diffusion:
            self.action_head = get_action_head(cfg, self.vla.llm_dim)

        # Check that the model contains the action un-normalization key
        assert cfg.unnorm_key in self.vla.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

        # Get Hugging Face processor
        self.processor = get_processor(cfg)

        # Get expected image dimensions
        self.resize_size = get_image_resize_size(cfg)

        # Set robot type and instruction
        self.robot_type = "R1Pro"
        self.instruction = "turn on radio"
        self.action_idx = 0
        self.max_action_len = 10

    def reset(self) -> None:
        # No stateful components to reset for now
        return None

    def _to_numpy_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        obs_numpy: Dict[str, Any] = {}
        for key, value in obs.items():
            if isinstance(value, torch.Tensor):
                obs_numpy[key] = value.detach().cpu().numpy()
            else:
                obs_numpy[key] = value
        return obs_numpy

    def _generate_prop_state(self, proprio: np.ndarray) -> np.ndarray:
        idx = PROPRIOCEPTION_INDICES[self.robot_type]
        qpos_list = [
            proprio[idx["base_qpos"]], # 3
            proprio[idx["trunk_qpos"]], # 4
            proprio[idx["arm_left_qpos"]], # 7
            proprio[idx["arm_right_qpos"]], # 7
            proprio[idx["gripper_left_qpos"]].sum(axis=-1, keepdims=True), # 1
            proprio[idx["gripper_right_qpos"]].sum(axis=-1, keepdims=True), # 1
        ]
        return np.concatenate(qpos_list, axis=0)

    def _process_behavior_obs(self, obs_numpy: Dict[str, Any]) -> Dict[str, Any]:
        # Extract images
        try:
            full_image = obs_numpy[ROBOT_CAMERA_NAMES[self.robot_type]["head"] + "::rgb"][:, :, :3] # (720, 720, 3)
            left_wrist_image = obs_numpy[ROBOT_CAMERA_NAMES[self.robot_type]["left_wrist"] + "::rgb"][:, :, :3] # (480, 480, 3)
            right_wrist_image = obs_numpy[ROBOT_CAMERA_NAMES[self.robot_type]["right_wrist"] + "::rgb"][:, :, :3] # (480, 480, 3)
            prop_state = self._generate_prop_state(obs_numpy["robot_r1::proprio"]) # (23, )
        except KeyError as e:
            logging.error(e)
            raise
        
        return {
            "full_image": resize_image_for_policy(full_image, self.resize_size), # resize images to training image size
            "left_wrist_image": resize_image_for_policy(left_wrist_image, self.resize_size),
            "right_wrist_image": resize_image_for_policy(right_wrist_image, self.resize_size),
            "state": prop_state,
            "instruction": self.instruction,
        }

    def act(self, obs: Dict[str, Any]) -> torch.Tensor:
        try:
            if self.action_idx % self.max_action_len == 0:
                # The websocket server converts incoming payload to torch tensors; convert back for OpenVLA utils
                obs_numpy = self._to_numpy_obs(obs)
                obs = self._process_behavior_obs(obs_numpy)
                action_list = get_vla_action(
                    self.cfg,
                    self.vla,
                    self.processor,
                    obs,
                    self.instruction,
                    action_head=self.action_head,
                    proprio_projector=self.proprio_projector,
                    use_film=self.cfg.use_film,
                )

                action_tensor = torch.from_numpy(np.array(action_list)).to(torch.float32)
                self.action_idx = 0
            
            action = action_tensor[self.action_idx]
            self.action_idx += 1            
            return action
        
        except:  # noqa: E722
            logging.error(traceback.format_exc())
            raise


@dataclass
class DeployConfig:
    # fmt: off

    # Server Configuration
    host: str = "0.0.0.0"                                               # Host IP Address
    port: int = 8000                                                    # Host Port

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    num_diffusion_steps_inference: int = 50          # (When `diffusion==True`) Number of diffusion steps used for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 3                     # Number of images in the VLA input (default: 3)
    use_proprio: bool = True                         # Whether to include proprio state in input
    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    lora_rank: int = 32                              # Rank of LoRA weight matrix (MAKE SURE THIS MATCHES TRAINING!)
    unnorm_key: Union[str, Path] = ""                # Action un-normalization key
    use_relative_actions: bool = False               # Whether to use relative actions (delta joint angles)
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # Utils
    #################################################################################################################
    seed: int = 42                                    # Random Seed (for reproducibility)
    # fmt: on


@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    policy = OpenVLAPolicy(cfg)
    server = WebsocketPolicyServer(
        policy=policy,
        host=cfg.host,
        port=cfg.port,
        metadata={"model_family": cfg.model_family},
    )
    server.serve_forever()


if __name__ == "__main__":
    deploy()
