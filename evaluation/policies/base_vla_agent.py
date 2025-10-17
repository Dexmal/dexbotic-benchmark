"""
Abstract Base Class for VLA Agent, Defines Unified Interface and Common Functions
"""

import json
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Optional
import numpy as np
import requests
from omegaconf import OmegaConf

from .adaptive_ensemble import AdaptiveEnsembler


class BaseVLAAgent(ABC):
    """
    Abstract Base Class for VLA Agent
    
    Attributes:
        base_url (str): Base URL for VLA service
        temperature (float): Generation temperature parameter
        replan_step (int): Replanning step count
        use_delta (bool): Whether to use delta mode
        current_step (int): Current step count
        last_act (Optional[np.ndarray]): Previous action
        action_queue (deque): Action queue
        action_ensembler (Optional[AdaptiveEnsembler]): Action ensemble
    """
    
    def __init__(self, config):
        """
        Initialize VLA agent
        
        Args:
            config: Loaded OmegaConf object
        """
        
        # Basic configuration
        self.base_url = config.base_url
        self.temperature = getattr(config, 'temperature', 1.0)
        self.replan_step = config.replan_step
        self.use_delta = getattr(config, 'use_delta', False)
        
        # State variables
        self.current_step = 0
        self.last_act = None
        self.action_queue = deque()
        
        # Action ensemble configuration
        self.action_ensemble_horizon = getattr(config, 'action_ensemble_horizon', 7)
        self.adaptive_ensemble_alpha = getattr(config, 'adaptive_ensemble_alpha', 0.1)
        
        # Initialize action ensemble
        if getattr(config, 'action_ensemble', False):
            self.action_ensembler = AdaptiveEnsembler(
                self.action_ensemble_horizon, 
                self.adaptive_ensemble_alpha
            )
        else:
            self.action_ensembler = None
            
        # Call subclass-specific initialization
        self._init_specific_config(config)

    @abstractmethod
    def _init_specific_config(self, config: OmegaConf) -> None:
        """
        Subclass-specific configuration initialization
        
        Args:
            config (OmegaConf): Configuration object
        """
        pass

    def reset(self) -> None:
        """
        Reset agent state
        
        Clear current step count, previous action, action queue, and ensemble state.
        """
        self.current_step = 0
        self.last_act = None
        self.action_queue.clear()
        
        if self.action_ensembler is not None:
            self.action_ensembler.reset()

    @abstractmethod
    def step(self, obs: Any, goal: str, episode_first_frame: bool = None) -> Any:
        """
        Execute one step of inference
        
        Args:
            obs: Environment observation
            goal (str): Goal description
            
        Returns:
            Predicted action
        """
        pass

    def _add_new_action(self, obs: Any, goal: str, episode_first_frame: bool) -> None:
        """
        Add new action to queue
        
        This method processes observation data, calls VLA service to get action predictions,
        and adds processed actions to the queue.
        
        Args:
            obs: Environment observation
            goal (str): Goal description
        """
        # Prepare state information
        state = self._prepare_state(obs)
        
        # Prepare image data
        images = self._prepare_images(obs)
        
        # Call VLA service
        raw_actions = self._call_vla_service(images, goal, state, episode_first_frame=episode_first_frame)
        
        # Process action predictions
        self._process_action_predictions(raw_actions)

    @abstractmethod
    def _prepare_state(self, obs: Any) -> np.ndarray:
        """
        Prepare state information
        
        Args:
            obs: Environment observation
            
        Returns:
            np.ndarray: Processed state information
        """
        pass

    @abstractmethod
    def _prepare_images(self, obs: Any) -> list:
        """
        Prepare image data
        
        Args:
            obs: Environment observation
            
        Returns:
            list: Encoded image list
        """
        pass

    def _call_vla_service(self, images: list, goal: str, state: np.ndarray, episode_first_frame: bool) -> np.ndarray:
        """
        Call VLA service to get action predictions
        
        Args:
            images (list): Encoded image list
            goal (str): Goal description
            state (np.ndarray): State information
            
        Returns:
            np.ndarray: Raw action predictions
            
        Raises:
            SystemExit: Exits program when VLA service does not return valid response
        """
        text = f'What action should the robot take to {goal}?'
        
        # Prepare request data (specific parameters determined by subclass)
        data = self._prepare_request_data(text, state, episode_first_frame=episode_first_frame)
        
    
        # Send request
        ret = requests.post(
            self.base_url + "/process_frame",
            data=data,
            files=[("image", img) for img in images],
        )
        
        # Check if request was successful
        ret.raise_for_status()
        
        # Parse response
        response_data = ret.json()
        response = response_data.get('response')
        
        # Check if response is valid
        if response is None:
            print(f"Error: VLA service did not return valid response. Response data: {response_data}")
            raise SystemExit("VLA service response invalid, exiting program")
            
        return response

    def _prepare_request_data(self, text: str, state: np.ndarray, episode_first_frame: bool) -> dict:
        """
        Prepare request data, subclasses can override this method to customize parameters
        
        Args:
            text (str): Request text
            state (np.ndarray): State information
            
        Returns:
            dict: Request data dictionary
        """
        data = {"text": text, "temperature": 0.8}
        
        # If state information exists, add to request
        if state is not None:
            data["states"] = json.dumps(state.tolist())

        if episode_first_frame is not None:
            data["episode_first_frame"] = episode_first_frame
            
        return data

    @abstractmethod
    def _process_action_predictions(self, raw_actions: list) -> None:
        """
        Process action predictions
        
        Perform ensemble processing on raw action predictions and generate actions
        for multiple time steps. Subclasses need to implement specific processing logic.
        
        Args:
            raw_actions (list): Raw action predictions
        """
        pass