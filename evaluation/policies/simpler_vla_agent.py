"""
VLA Agent Implementation for Simpler Project
"""

from typing import Optional, Sequence, Tuple, Dict
import numpy as np
import cv2
import matplotlib.pyplot as plt
from transforms3d.euler import euler2axangle
from omegaconf import OmegaConf

from .base_vla_agent import BaseVLAAgent


class SimplerVLAAgent(BaseVLAAgent):
    """
    VLA Agent Implementation for Simpler Project
    
    Attributes:
        image_size (tuple): Image size
    """
    
    def _init_specific_config(self, config: OmegaConf) -> None:
        """
        Initialize Simpler-specific configuration
        
        Args:
            config (OmegaConf): Configuration object
        """
        # Simpler-specific configuration
        self.image_size = getattr(config, 'image_size', (224, 224))
        
        # Ensure delta mode is used
        assert self.use_delta, "SimplerVLAAgent only supports delta mode"

    def reset(self, task_description: Optional[str] = None) -> None:
        """
        Reset agent state
        
        Args:
            task_description (Optional[str]): Task description, reset state if different from previous
        """
        super().reset()

    def step(self, image: np.ndarray, task_description: Optional[str] = None, episode_first_frame: bool=None) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Execute one step of inference
        
        Args:
            image (np.ndarray): Input image with shape (H, W, 3), uint8 format
            task_description (Optional[str]): Task description, reset policy state if different from previous
            
        Returns:
            Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]: 
                - raw_action: Raw policy action output
                - action: Processed action sent to Maniskill2 environment
        """
        # If action queue is empty, add new action
        if len(self.action_queue) == 0:
            self._add_new_action(image, task_description, episode_first_frame=episode_first_frame)
        
        # Pop action from queue
        action, raw_action = self.action_queue.popleft()

        return raw_action, action

    def _prepare_state(self, obs: np.ndarray) -> np.ndarray:
        """
        Prepare Simpler environment state information
        
        Args:
            obs (np.ndarray): Input image
            
        Returns:
            np.ndarray: State information (Simpler environment does not need state information)
        """
        # Simpler environment does not need state information
        return None

    def _prepare_images(self, obs: np.ndarray) -> list:
        """
        Prepare Simpler environment image data
        
        Args:
            obs (np.ndarray): Input image
            
        Returns:
            list: Encoded image list (numpy arrays)
        """
        # Resize image
        image = self._resize_image(obs)
        # Convert to BGR format
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Encode as PNG format (Simpler project directly uses numpy arrays)
        _, encoded_image = cv2.imencode('.png', image)
        
        return [encoded_image]

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            np.ndarray: Resized image
        """
        image = cv2.resize(image, tuple(self.image_size), interpolation=cv2.INTER_AREA)
        return image

    def _process_action_predictions(self, raw_actions: list) -> None:
        """
        Process Simpler environment action predictions
        
        Args:
            raw_actions (list): Raw action predictions
        """
        raw_actions = np.array(raw_actions)
        # Apply action ensemble
        if self.action_ensembler is not None:
            raw_actions = self.action_ensembler.ensemble_action(raw_actions)[None]
        # Generate actions for multiple time steps
        for i in range(self.replan_step):
            # Process gripper action
            gripper = 2.0 * (raw_actions[i][6] > 0.5) - 1.0  # Convert to [-1, 1] range
            
            # Process rotation action
            rotation = raw_actions[i][3:6]
            axes, angles = euler2axangle(*rotation)
            action_rotation_axangle = axes * angles
            
            # Build action dictionary
            action = {
                'world_vector': np.array(raw_actions[i][:3]),
                'rot_axangle': np.array(action_rotation_axangle),
                'gripper': np.array([gripper]),
                'terminate_episode': np.zeros(1),
            }
            
            # Build raw action dictionary
            raw_action = {
                "world_vector": np.array(raw_actions[i][:3]),
                "rotation_delta": np.array(raw_actions[i][3:6]),
                "open_gripper": action['gripper'],
            }
            
            # Add to queue
            self.action_queue.append((action, raw_action))

    def visualize_epoch(self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str) -> None:
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        # set up plt figure
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # plot actions
        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)