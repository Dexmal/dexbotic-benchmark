#!/usr/bin/env python3
"""
Simpler Environment Evaluator

Simpler ManiSkill2 environment evaluator implemented based on BaseEvaluator.
Supports multiple policy model types including RT1, Octo, VLA, etc.

Usage:
from evaluation.simpler.simple_evaluator import SimpleEvaluator
from omegaconf import OmegaConf

config = OmegaConf.load("config.yaml")
evaluator = SimpleEvaluator(config)
results = evaluator.run_evaluation()
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from omegaconf import OmegaConf

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from evaluation.evaluator.base_evaluator import BaseEvaluator
from evaluation.utils.simpler_maniskill2 import maniskill2_evaluator

logger = logging.getLogger(__name__)


class SimplerEvaluator(BaseEvaluator):
    """
    Simpler Environment Evaluator
    """
    
    def __init__(self, config: OmegaConf, output_structure: Dict[str, Path]):
        """
        Initialize Simpler evaluator
        
        Args:
            config: OmegaConf configuration object
            output_structure: Dictionary containing output structure
        """
        # Call parent class initialization first so self.config is set
        super().__init__(config, output_structure)
        
        # Set environment variables
        self._setup_environment_variables()
        logger.info("Simpler evaluator initialization completed")
    
    def _setup_environment_variables(self):
        """
        Setup environment variables
        
        Configure necessary environment variables and system settings
        """
        # Prevent display issues
        os.environ["DISPLAY"] = ""
        # Prevent JAX processes from occupying all GPU memory
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        
        logger.info("Environment variables setup completed")
    
    def setup_environment(self) -> Any:
        return None
    
    def setup_model(self) -> Any:
        """
        Setup evaluation model
        
        Create corresponding policy model instance based on configuration parameters
        
        Returns:
            Any: Policy model object
            
        Raises:
            NotImplementedError: Unsupported model type
            ImportError: Model import failed
        """
        policy_model = self.config.policy_model
        logger.info(f"Creating policy model: {policy_model}")
        
        try:
            if policy_model == "rt1":
                if not self.config.get("ckpt_path"):
                    raise ValueError("RT1 model requires checkpoint path to be specified")
                
                from simpler.simpler_env.policies.rt1.rt1_model import RT1Inference
                model = RT1Inference(
                    saved_model_path=self.config.ckpt_path,
                    policy_setup=self.config.get("policy_setup", "google_robot"),
                    action_scale=self.config.get("action_scale", 1.0),
                )
                
            elif "octo" in policy_model:
                ckpt_path = self.config.get("ckpt_path")
                if not ckpt_path or ckpt_path == "None":
                    ckpt_path = policy_model
                    
                if "server" in policy_model:
                    from simpler.simpler_env.policies.octo.octo_server_model import OctoServerInference
                    model = OctoServerInference(
                        model_type=ckpt_path,
                        policy_setup=self.config.get("policy_setup", "google_robot"),
                        action_scale=self.config.get("action_scale", 1.0),
                    )
                else:
                    from simpler.simpler_env.policies.octo.octo_model import OctoInference
                    model = OctoInference(
                        model_type=ckpt_path,
                        policy_setup=self.config.get("policy_setup", "google_robot"),
                        init_rng=self.config.get("octo_init_rng", 0),
                        action_scale=self.config.get("action_scale", 1.0),
                    )
                    
            elif policy_model == "vla":
                from evaluation.policies.simpler_vla_agent import SimplerVLAAgent
                model = SimplerVLAAgent(self.config)
                
            else:
                raise NotImplementedError(f"Unsupported policy model type: {policy_model}")
            
            logger.info(f"Policy model created successfully: {policy_model}")
            return model
            
        except ImportError as e:
            logger.error(f"Failed to import policy model: {e}")
            raise
        except Exception as e:
            logger.error(f"Error occurred while creating policy model: {e}")
            raise
    
    def _run_evaluation_impl(self) -> Dict[str, Any]:
        """
        Implement specific evaluation logic
        
        Use maniskill2_evaluator to execute evaluation and collect results
        
        Returns:
            Dict[str, Any]: Evaluation results
        """
        logger.info("Starting Simpler environment evaluation...")
        
        # Prepare evaluation arguments
        eval_args = self._prepare_evaluation_args()
        # Execute evaluation
        success_arr = maniskill2_evaluator(self.model, eval_args)

        # Calculate statistical results
        total_episodes = len(success_arr)
        successful_episodes = sum(success_arr)
        success_rate = successful_episodes / total_episodes if total_episodes > 0 else 0.0
        
        # Build results dictionary
        results = {
            "success_array": success_arr,
            "total_episodes": total_episodes,
            "successful_episodes": successful_episodes,
            "success_rate": success_rate,
            "model_type": self.config.policy_model,
            "env_name": self.config.env_name,
            "robot_type": self.config.robot,
            "scene_name": self.config.get("scene_name", "default"),
            "config": OmegaConf.to_container(self.config, resolve=True),
        }
        
        logger.info(f"Evaluation completed - Total tasks: {total_episodes}, Success rate: {success_rate:.2%}")
        
        return results
    
    def _prepare_evaluation_args(self):
        """
        Prepare evaluation arguments
        
        Convert configuration to parameter format required by maniskill2_evaluator
        
        Returns:
            argparse.Namespace: Evaluation arguments object
        """
        # Create a simple arguments object
        class Args:
            def __init__(self, config, output_structure):
                # Basic parameters
                self.policy_model = config.policy_model
                self.ckpt_path = config.get("ckpt_path")
                self.env_name = config.env_name
                self.scene_name = config.get("scene_name", "google_pick_coke_can_1_v4")
                self.robot = config.robot
                self.enable_raytracing = config.get("enable_raytracing", False)
                
                # Observation parameters
                self.obs_camera_name = config.get("obs_camera_name")
                
                # Control parameters
                self.control_freq = config.get("control_freq", 3)
                self.sim_freq = config.get("sim_freq", 513)
                self.max_episode_steps = config.get("max_episode_steps", 80)
                self.rgb_overlay_path = config.get("rgb_overlay_path")
                
                # Robot initial position parameters
                self.robot_init_x_range = config.get("robot_init_x_range", [0.35, 0.35, 1])
                self.robot_init_y_range = config.get("robot_init_y_range", [0.20, 0.20, 1])
                self.robot_init_rot_quat_center = config.get("robot_init_rot_quat_center", [1, 0, 0, 0])
                self.robot_init_rot_rpy_range = config.get("robot_init_rot_rpy_range", [0, 0, 1, 0, 0, 1, 0, 0, 1])
                
                # Object variation parameters
                self.obj_variation_mode = config.get("obj_variation_mode", "xy")
                self.obj_episode_range = config.get("obj_episode_range", [0, 60])
                self.obj_init_x_range = config.get("obj_init_x_range", [-0.35, -0.12, 5])
                self.obj_init_y_range = config.get("obj_init_y_range", [-0.02, 0.42, 5])
                
                # Other parameters
                self.additional_env_save_tags = config.get("additional_env_save_tags")
                self.additional_env_build_kwargs = config.get("additional_env_build_kwargs", {})
                self.octo_init_rng = config.get("octo_init_rng", 0)

                self.logging_dir = output_structure["logs_dir"]
                self.video_path = output_structure["videos_dir"]

                # Process robot position parameters
                self._process_robot_position_args()
                self._process_object_position_args()
            
            def _process_robot_position_args(self):
                """
                Process robot position parameters
                
                Convert position ranges in configuration to specific coordinate arrays
                """
                from transforms3d.euler import euler2quat
                from sapien.core import Pose
                
                # Process robot initial position
                self.robot_init_xs = np.linspace(
                    self.robot_init_x_range[0], 
                    self.robot_init_x_range[1], 
                    int(self.robot_init_x_range[2])
                )
                self.robot_init_ys = np.linspace(
                    self.robot_init_y_range[0], 
                    self.robot_init_y_range[1], 
                    int(self.robot_init_y_range[2])
                )
                
                # Generate robot initial rotation quaternions
                self.robot_init_quats = []
                r_range = np.linspace(
                    self.robot_init_rot_rpy_range[0], 
                    self.robot_init_rot_rpy_range[1], 
                    int(self.robot_init_rot_rpy_range[2])
                )
                p_range = np.linspace(
                    self.robot_init_rot_rpy_range[3], 
                    self.robot_init_rot_rpy_range[4], 
                    int(self.robot_init_rot_rpy_range[5])
                )
                y_range = np.linspace(
                    self.robot_init_rot_rpy_range[6], 
                    self.robot_init_rot_rpy_range[7], 
                    int(self.robot_init_rot_rpy_range[8])
                )
                
                for r in r_range:
                    for p in p_range:
                        for y in y_range:
                            self.robot_init_quats.append(
                                (Pose(q=euler2quat(r, p, y)) * Pose(q=self.robot_init_rot_quat_center)).q
                            )
            
            def _process_object_position_args(self):
                """
                Process object position parameters
                
                Convert object position ranges in configuration to specific coordinate arrays
                """
                if self.obj_variation_mode == "xy":
                    self.obj_init_xs = np.linspace(
                        self.obj_init_x_range[0], 
                        self.obj_init_x_range[1], 
                        int(self.obj_init_x_range[2])
                    )
                    self.obj_init_ys = np.linspace(
                        self.obj_init_y_range[0], 
                        self.obj_init_y_range[1], 
                        int(self.obj_init_y_range[2])
                    )
        
        args = Args(self.config, self.output_structure)
        return args