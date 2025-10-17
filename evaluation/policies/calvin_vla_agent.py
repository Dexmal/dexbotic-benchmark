"""
VLA Agent Implementation for Calvin Project
"""

import numpy as np
import torch
import math
import cv2
import json
from omegaconf import OmegaConf

from .base_vla_agent import BaseVLAAgent


class CalvinVLAAgent(BaseVLAAgent):
    """
    VLA Agent Implementation for Calvin Project
    
    Attributes:
        rgb_cameras (list): RGB camera list
        depth_cameras (list): Depth camera list
        send_images (list): List of images to send
    """
    
    def _init_specific_config(self, config: OmegaConf) -> None:
        """
        Initialize Calvin-specific configuration
        
        Args:
            config (OmegaConf): Configuration object
        """
        # Calvin-specific configuration
        self.rgb_cameras = config.datamodule.observation_space.rgb_obs
        self.depth_cameras = config.datamodule.observation_space.depth_obs
        self.send_images = config.send_image

    def set_init_action(self, raw_action: torch.Tensor) -> None:
        """
        Set initial action
        
        Args:
            raw_action (torch.Tensor): Raw action tensor
        """
        # Extract position and gripper actions
        action = torch.cat([raw_action[0:6], raw_action[14:15]]).cpu().numpy()
        self.last_act = action

    def step(self, obs: dict, goal: str, episode_first_frame: bool=None) -> torch.Tensor:
        """
        Execute one step of inference
        
        Args:
            obs (dict): Calvin environment observation
                - rgb_obs: RGB image observation
                - depth_obs: Depth image observation
                - robot_obs_raw: Robot raw observation
                - task: Task name
            goal (str): Language goal
            
        Returns:
            torch.Tensor: Predicted action with shape [1, 1, 7]
        """
            
        # If action queue is empty, add new action
        if len(self.action_queue) == 0:
            self._add_new_action(obs, goal, episode_first_frame=episode_first_frame)
            
        # Pop action from queue
        action = self.action_queue.popleft()
        self.last_act = action

        # Convert to tensor format
        action = torch.tensor(action).unsqueeze(0).unsqueeze(0).to(obs['robot_obs_raw'].device)
        self.current_step += 1

        return action

    def _prepare_state(self, obs: dict) -> np.ndarray:
        """
        Prepare Calvin environment state information
        
        Args:
            obs (dict): Calvin environment observation
            
        Returns:
            np.ndarray: Processed state information
        """
        # Extract robot state information
        state = obs['robot_obs_raw'].cpu().numpy()
        # Keep only position and gripper information
        state = np.concatenate([state[0:6], state[14:15]])
        
        # If no initial action yet, set initial action
        if self.last_act is None:
            self.set_init_action(obs['robot_obs_raw'])
            
        return state

    def _prepare_images(self, obs: dict) -> list:
        """
        Prepare Calvin environment image data
        
        Args:
            obs (dict): Calvin environment observation
            
        Returns:
            list: Encoded image list (numpy arrays)
        """
        # Convert observation to image format
        images = self._obs2image(obs)
        # Select images to send
        images = [images[k] for k in self.send_images]
        
        # Encode images (uniformly use numpy arrays)
        encoded_images = []
        for image in images:
            # Convert to BGR format
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # Encode as PNG format
            ret, encoded_image = cv2.imencode('.png', image)
            # Uniformly use numpy arrays, do not convert to bytes
            encoded_images.append(encoded_image.tobytes())
            
        return encoded_images

    def _obs2image(self, obs: dict) -> dict:
        """
        Convert Calvin observation to image format
        
        Args:
            obs (dict): Calvin environment observation
            
        Returns:
            dict: Image dictionary
        """
        images = {}

        # Process RGB images
        for camera in self.rgb_cameras:
            # Convert from tensor to numpy array and adjust value range to [0, 255]
            rgb_image = ((obs['rgb_obs'][camera][0,0].permute(1,2,0).cpu() + 1) / 2 * 255).numpy().astype('uint8')
            images[camera] = rgb_image
            
        return images

    def _process_single_action(self, last_action: np.ndarray, 
                              predict_action: np.ndarray) -> np.ndarray:
        """
        Process single Calvin action
        
        Args:
            last_action (np.ndarray): Previous action
            predict_action (np.ndarray): Predicted action
            
        Returns:
            np.ndarray: Processed action
        """
        if self.use_delta:
            # Delta mode: add predicted action to previous action
            original_action = np.copy(last_action)
            original_action[6:] = 0  # Reset gripper action
            action = original_action + predict_action
        else:
            # Absolute mode: use predicted action directly
            action = predict_action
            
        # Process gripper action
        if action[-1] > 0:
            action[-1] = 1
        else:
            action[-1] = -1
            
        # Normalize angles
        action = self._normalize_angles(action)
        
        return action

    def _process_action_predictions(self, raw_actions: list) -> None:
        """
        Process action predictions
        
        Args:
            raw_actions (list): Raw action predictions
        """
        raw_actions = np.array(raw_actions)
        if self.action_ensembler is not None:
            raw_actions = self.action_ensembler.ensemble_action(raw_actions)[None]
        last_act = self.last_act
        for i in range(self.replan_step):
            action = self._process_single_action(last_act, raw_actions[i])
            last_act = action
            self.action_queue.append(action)

    def _prepare_request_data(self, text: str, state: np.ndarray, episode_first_frame: bool) -> dict:
        """
        Prepare Calvin environment request data, including temperature parameter
        
        Args:
            text (str): Request text
            state (np.ndarray): State information
            
        Returns:
            dict: Request data dictionary
        """
        data = {"text": text, "temperature": self.temperature, "episode_first_frame": episode_first_frame}
        
        # If state information exists, add to request
        if state is not None:
            data["states"] = json.dumps(state.tolist())
            
        return data

    def _normalize_angles(self, action: np.ndarray) -> np.ndarray:
        """
        Normalize angle values to [-π, π] range
        
        Args:
            action (np.ndarray): Action array containing angles
            
        Returns:
            np.ndarray: Normalized action
        """
        # Normalize angles to [-π, π] range
        action[3:6] = np.where(
            action[3:6] > math.pi,
            action[3:6] - 2 * math.pi,
            action[3:6]
        )
        action[3:6] = np.where(
            action[3:6] < -math.pi,
            action[3:6] + 2 * math.pi,
            action[3:6]
        )
        return action