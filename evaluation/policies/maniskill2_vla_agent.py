"""
VLA Agent Implementation for ManiSkill2 Environment
"""

import numpy as np
import cv2
from typing import Any, Optional
from omegaconf import OmegaConf
from maniskill2_learn.utils.data import GDict
from .base_vla_agent import BaseVLAAgent


class ManiSkill2VLAAgent(BaseVLAAgent):
    """
    VLA Agent Implementation for ManiSkill2 Environment

    Attributes:
        image_size (tuple): Image size for processing
        enabled_cameras (list): List of camera names to use for image processing
    """
    
    def _init_specific_config(self, config: OmegaConf) -> None:
        """
        Initialize ManiSkill2-specific configuration

        Args:
            config (OmegaConf): Configuration object
        """
        # ManiSkill2-specific configuration
        self.image_size = getattr(config, 'image_size', (336, 336))
        self.enabled_cameras = getattr(config, 'enabled_cameras', ['base_camera', 'hand_camera'])

        # Ensure delta mode is used
        assert self.use_delta, "ManiSkill2VLAAgent only supports delta mode"

    def step(self, obs: Any, goal: Optional[str] = None, episode_first_frame: Optional[bool] = None) -> np.ndarray:
        """
        Execute one step of inference

        Args:
            obs: Environment observation
            goal (Optional[str]): Goal description
            episode_first_frame (bool): Whether this is the first frame of the episode

        Returns:
            np.ndarray: Action array
        """
        # If action queue is empty, add new action
        if len(self.action_queue) == 0:
            self._add_new_action(obs, goal, episode_first_frame=episode_first_frame)

        # Pop action from queue
        action = self.action_queue.popleft()

        return action

    def _prepare_state(self, obs: Any) -> np.ndarray:
        """
        Prepare ManiSkill2 environment state information

        Args:
            obs: Environment observation

        Returns:
            np.ndarray: State information
        """
        # Extract state information from observation
        if isinstance(obs, dict) and 'extra' in obs:
            # For ManiSkill2 observations with extra data
            extra_data = obs['extra']

            # Combine tcp_pose, gripper_state, and goal_pos into a single list
            state_features = []

            if 'tcp_pose' in extra_data:
                state_features.extend(extra_data['tcp_pose'])

            if 'gripper_state' in extra_data:
                state_features.append(extra_data['gripper_state'])

            if 'goal_pos' in extra_data:
                state_features.extend(extra_data['goal_pos'])
            else:
                state_features.extend([0.0, 0.0, 0.0])

            return np.array(state_features, dtype=np.float32)

        elif isinstance(obs, dict) and 'agent' in obs:
            # For state observations - extract proprioception data
            agent_data = obs['agent']

            # Combine proprioception features: qpos, qvel, base_pose
            state_features = []

            if 'qpos' in agent_data:
                state_features.extend(agent_data['qpos'])

            if 'qvel' in agent_data:
                state_features.extend(agent_data['qvel'])

            if 'base_pose' in agent_data:
                state_features.extend(agent_data['base_pose'])

            return np.array(state_features, dtype=np.float32)

        elif isinstance(obs, np.ndarray):
            # For direct state arrays
            return obs
        else:
            # For other observation types, return None
            return None

    def _prepare_images(self, obs: Any) -> list:
        """
        Prepare ManiSkill2 environment image data

        Args:
            obs: Environment observation

        Returns:
            list: Encoded image list
        """
        images = []

        if isinstance(obs, dict) and 'image' in obs:
            # Handle ManiSkill2 observation format with image dictionary
            obs_gdict = GDict(obs)

            # Check if configured cameras exist in observation
            available_cameras = list(obs_gdict['image'].keys())
            missing_cameras = [cam for cam in self.enabled_cameras if cam not in available_cameras]

            if missing_cameras:
                raise ValueError(f"Configured cameras {missing_cameras} not found in observation. "
                               f"Available cameras: {available_cameras}")

            # Process enabled cameras based on configuration
            for camera_name in self.enabled_cameras:
                camera_data = obs_gdict['image'][camera_name]
                if 'rgb' in camera_data:
                    rgb_image = camera_data['rgb']
                    processed_image = self._process_image(rgb_image)
                    images.append(processed_image)
        elif isinstance(obs, dict):
            # Handle dictionary observations (rgbd, rgb, etc.)
            for key in obs:
                if key.startswith('rgb') or key.startswith('image'):
                    image_data = obs[key]
                    if isinstance(image_data, np.ndarray) and len(image_data.shape) == 3:
                        # Process RGB image
                        processed_image = self._process_image(image_data)
                        images.append(processed_image)
        elif isinstance(obs, np.ndarray) and len(obs.shape) == 3:
            # Handle direct image observations
            processed_image = self._process_image(obs)
            images.append(processed_image)

        return images

    def _process_image(self, image: np.ndarray) -> bytes:
        """
        Process and encode image

        Args:
            image (np.ndarray): Input image

        Returns:
            bytes: Encoded image bytes
        """
        # Resize image to configured size
        image = cv2.resize(image, self.image_size)

        # Convert to BGR format as in the original example
        if image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Encode as JPG
        _, encoded_image = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])

        return encoded_image.tobytes()

    def _process_action_predictions(self, raw_actions: list) -> None:
        """
        Process ManiSkill2 environment action predictions

        Args:
            raw_actions (list): Raw action predictions
        """
        raw_actions = np.array(raw_actions)

        # Apply action ensemble
        if self.action_ensembler is not None:
            raw_actions = self.action_ensembler.ensemble_action(raw_actions)

        # Ensure raw_actions is 2D for consistent processing
        if raw_actions.ndim == 1:
            raw_actions = raw_actions[None, :]

        # Apply gripper binarization
        is_zero = (raw_actions[:, -1] < 0.5)
        raw_actions[is_zero, -1] = 1
        raw_actions[~is_zero, -1] = -1

        # Generate actions for multiple time steps
        for i in range(min(self.replan_step, len(raw_actions))):
            # Extract action for current time step
            if len(raw_actions.shape) > 1:
                action = raw_actions[i]
            else:
                action = raw_actions
            
            # Ensure action is a numpy array
            if not isinstance(action, np.ndarray):
                action = np.array(action)
            
            # Add to queue
            self.action_queue.append(action)