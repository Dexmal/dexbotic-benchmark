"""
Adaptive Ensemble Class for Action Prediction Integration and Optimization
"""

from collections import deque
import numpy as np
from abc import ABC, abstractmethod


class AdaptiveEnsembler:
    """
    Adaptive Ensemble for Action Prediction Integration
    
    Attributes:
        pred_action_horizon (int): Time horizon for action prediction
        action_history (deque): Queue storing historical action predictions
        adaptive_ensemble_alpha (float): Weight parameter for adaptive ensemble
    """
    
    def __init__(self, pred_action_horizon: int, adaptive_ensemble_alpha: float = 0.0):
        """
        Initialize adaptive ensemble
        
        Args:
            pred_action_horizon (int): Time horizon for action prediction
            adaptive_ensemble_alpha (float): Weight parameter for adaptive ensemble, default 0.0
        """
        self.pred_action_horizon = pred_action_horizon
        self.action_history = deque(maxlen=self.pred_action_horizon)
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha

    def reset(self) -> None:
        """
        Reset ensemble state and clear historical action records
        """
        self.action_history.clear()

    def ensemble_action(self, cur_action: np.ndarray) -> np.ndarray:
        """
        Perform ensemble processing on current action
        
        Args:
            cur_action (np.ndarray): Current action prediction
            
        Returns:
            np.ndarray: Ensemble action prediction
        """
        # Add current action to historical records
        self.action_history.append(cur_action)
        num_actions = len(self.action_history)
        
        # Build prediction matrix based on action dimensions
        if cur_action.ndim == 1:
            curr_act_preds = np.stack(self.action_history)
        else:
            curr_act_preds = np.stack(
                [pred_actions[i] for (i, pred_actions) in zip(range(num_actions - 1, -1, -1), self.action_history)]
            )

        # Calculate cosine similarity between current prediction and all historical predictions
        ref = curr_act_preds[num_actions-1, :]
        previous_pred = curr_act_preds
        dot_product = np.sum(previous_pred * ref, axis=1)  
        norm_previous_pred = np.linalg.norm(previous_pred, axis=1)  
        norm_ref = np.linalg.norm(ref)  
        cos_similarity = dot_product / (norm_previous_pred * norm_ref + 1e-7)

        # Calculate weights for each prediction based on cosine similarity
        weights = np.exp(self.adaptive_ensemble_alpha * cos_similarity)
        weights = weights / weights.sum()
  
        # Calculate weighted average action prediction
        cur_action = np.sum(weights[:, None] * curr_act_preds, axis=0)

        return cur_action
