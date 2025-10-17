"""
VLA Agent Implementation for Libero Project
"""

import numpy as np
import cv2
import requests
import json
from omegaconf import OmegaConf
from typing import Optional

from .base_vla_agent import BaseVLAAgent


class LiberoVLAAgent(BaseVLAAgent):
    """
    VLA Agent Implementation for Libero Project
  
    Attributes:
        replan_step (int): Replanning step count
    """
    
    def _init_specific_config(self, config: OmegaConf) -> None:
        """
        Initialize Libero-specific configuration
        
        Args:
            config (OmegaConf): Configuration object
        """
        return

    def step(self, obs: dict, goal: str, episode_first_frame: bool=None) -> np.ndarray:
        """
        Execute one step of inference
        
        Args:
            obs (dict): Libero environment observation containing 'image' key
            goal (str): Task description
            
        Returns:
            np.ndarray: Predicted action
        """
        # If action queue is empty, add new action
        if len(self.action_queue) == 0:
            self._add_new_action(obs, goal, episode_first_frame=episode_first_frame)
        
        # Pop action from queue
        action = self.action_queue.popleft()
        self.last_act = action
        
        return action

    def _prepare_state(self, obs: dict) -> np.ndarray:
        """
        Prepare Libero environment state information
        
        Args:
            obs (dict): Libero environment observation
            
        Returns:
            np.ndarray: State information (Libero environment does not need state information)
        """
        # Libero environment does not need state information
        return None

    def _prepare_images(self, obs: dict) -> list:
        """
        Prepare Libero environment image data
        
        Args:
            obs (dict): Libero environment observation containing 'image' key
            
        Returns:
            list: Encoded image list (byte format)
        """
        # Get image
        images = [obs['image']]
        
        encoded_images = []
        for image in images:
            # Convert to BGR format
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # Encode as PNG format
            ret, encoded_image = cv2.imencode('.png', image)
            # Convert to byte format
            encoded_images.append(encoded_image.tobytes())
        
        return encoded_images

    def _process_action_predictions(self, raw_actions: np.ndarray) -> None:
        """
        Process Libero environment action predictions
        
        Args:
            raw_actions (np.ndarray): Raw action predictions
        """
        
        # Generate actions for multiple time steps
        for action in raw_actions[:self.replan_step]:
            gripper_action = -1 if action[-1] < 0 else 1
            self.action_queue.append(action[:-1] + [gripper_action])    
        