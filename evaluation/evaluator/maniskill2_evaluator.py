#!/usr/bin/env python3
"""
ManiSkill2 Environment Evaluator

ManiSkill2 environment evaluator implemented based on BaseEvaluator.
Supports the following tasks:
- PickCube-v0
- StackCube-v0  
- PickSingleYCB-v0
- PickSingleEGAD-v0
- PickClutterYCB-v0

Usage:
from evaluation.evaluator.maniskill2_evaluator import ManiSkill2Evaluator
from omegaconf import OmegaConf

config = OmegaConf.load("config.yaml")
evaluator = ManiSkill2Evaluator(config)
results = evaluator.run_evaluation()
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from omegaconf import OmegaConf
from mani_skill2.envs.sapien_env import BaseEnv

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from evaluation.evaluator.base_evaluator import BaseEvaluator
from evaluation.policies.maniskill2_vla_agent import ManiSkill2VLAAgent

logger = logging.getLogger(__name__)


class ManiSkill2Policy:
    """
    Base policy for ManiSkill2 evaluation
    """
    
    def __init__(self, env_id: str, observation_space, action_space):
        self.env_id = env_id
        self.observation_space = observation_space
        self.action_space = action_space
    
    def reset(self, observations):
        """Called at the beginning of an episode."""
        pass
    
    def act(self, observations) -> np.ndarray:
        """Act based on the observations."""
        raise NotImplementedError
    
    @classmethod
    def get_obs_mode(cls, env_id: str) -> str:
        """Get the observation mode for the policy."""
        return "rgbd"  # Use rgbd observations to include visual information
    
    @classmethod
    def get_control_mode(cls, env_id: str) -> str:
        """Get the control mode for the policy."""
        return "pd_ee_delta_pose"  # Default control mode


class RandomManiSkill2Policy(ManiSkill2Policy):
    """Random policy for ManiSkill2 evaluation"""
    
    def act(self, observations):
        return self.action_space.sample()
    
    @classmethod
    def get_obs_mode(cls, env_id: str) -> str:
        return "rgbd"  # Use state observations for random policy
    
    @classmethod
    def get_control_mode(cls, env_id: str) -> str:
        return "pd_ee_delta_pose"  # Same control mode as default


class DefaultManiSkill2Policy(ManiSkill2Policy):
    """Default policy that returns zero actions"""
    
    def act(self, observations):
        return np.zeros(self.action_space.shape, dtype=np.float32)


class ManiSkill2Evaluator(BaseEvaluator):
    """
    ManiSkill2 Environment Evaluator
    """
    
    def __init__(self, config: OmegaConf, output_structure: Dict[str, Path]):
        """
        Initialize ManiSkill2 evaluator
        
        Args:
            config: OmegaConf configuration object
            output_structure: Dictionary containing output structure
        """
        # Call parent class initialization first so self.config is set
        super().__init__(config, output_structure)
        
        # Set environment variables
        self._setup_environment_variables()
        logger.info("ManiSkill2 evaluator initialization completed")
        self.gripper_state = -1.0
    
    def _setup_environment_variables(self):
        """
        Setup environment variables
        
        Configure necessary environment variables and system settings
        """
        # Prevent display issues
        os.environ["DISPLAY"] = ""
        
        logger.info("Environment variables setup completed")
    
    def setup_environment(self) -> Any:
        """
        Setup ManiSkill2 environment
        
        Returns:
            Any: ManiSkill2 environment object
        """
        try:
            import gymnasium as gym
            import mani_skill2.envs  # Import ManiSkill2 environments to register them
            
            env_name = self.config.env_name
            policy_type = self.config.policy_type
            
            logger.info(f"Creating ManiSkill2 environment: {env_name}")
            
            # Determine observation and control modes based on policy type
            if policy_type == "random":
                obs_mode = RandomManiSkill2Policy.get_obs_mode(env_name)
                control_mode = RandomManiSkill2Policy.get_control_mode(env_name)
            else:
                obs_mode = DefaultManiSkill2Policy.get_obs_mode(env_name)
                control_mode = DefaultManiSkill2Policy.get_control_mode(env_name)
            
            logger.info(f"Observation mode: {obs_mode}, Control mode: {control_mode}")
            
            # Create environment using gymnasium
            env_kwargs = {}
            if obs_mode:
                env_kwargs["obs_mode"] = obs_mode
            if control_mode:
                env_kwargs["control_mode"] = control_mode
            
            # Set render mode (default to "cameras" as in the original evaluator)
            render_mode = self.config.get("render_mode", "cameras")
            env_kwargs["render_mode"] = render_mode
            env_kwargs["camera_cfgs"] = dict(width=256, height=256)
            max_episode_steps = self.config.get("max_episode_steps", 1000)
            env_kwargs["max_episode_steps"] = max_episode_steps
            
            env: BaseEnv = gym.make(env_name, **env_kwargs)
            
            logger.info("ManiSkill2 environment setup completed")
            return env
            
        except ImportError as e:
            logger.error("Failed to import gymnasium or ManiSkill2. Please ensure dependencies are installed.")
            raise
        except Exception as e:
            logger.error(f"Failed to setup ManiSkill2 environment: {e}")
            raise
    
    def setup_model(self) -> Any:
        """
        Setup evaluation model/policy
        
        Returns:
            Any: Policy model object
        """
        policy_type = self.config.policy_type
        env_name = self.config.env_name
        
        logger.info(f"Setting up policy: {policy_type}")
        
        if policy_type == "default":
            policy_cls = DefaultManiSkill2Policy
            model = policy_cls(env_name, self.env.observation_space, self.env.action_space)
        elif policy_type == "random":
            policy_cls = RandomManiSkill2Policy
            model = policy_cls(env_name, self.env.observation_space, self.env.action_space)
        elif policy_type == "vla":
            # Check if base_url is provided for VLA agent
            if not hasattr(self.config, 'base_url') or not self.config.base_url:
                raise ValueError("VLA policy requires base_url to be specified in configuration")
            
            model = ManiSkill2VLAAgent(self.config)
        else:
            raise NotImplementedError(f"Policy type {policy_type} not implemented")
        
        logger.info("Policy setup completed")
        return model
    
    def _run_evaluation_impl(self) -> Dict[str, Any]:
        """
        Implement ManiSkill2 evaluation logic
        
        Returns:
            Dict[str, Any]: Evaluation results
        """
        num_episodes = self.config.num_episodes
        render = self.config.get("render", False)
        
        logger.info(f"Starting evaluation with {num_episodes} episodes")
        
        results = {
            "environment": self.config.env_name,
            "num_episodes": num_episodes,
            "success_rate": 0.0,
            "episode_rewards": [],
            "episode_lengths": [],
            "successes": [],
            "episode_details": []
        }
        
        # Run evaluation episodes
        for episode in range(num_episodes):
            episode_result = self._run_single_episode(episode, render)
            results["episode_rewards"].append(episode_result["total_reward"])
            results["episode_lengths"].append(episode_result["episode_length"])
            results["successes"].append(episode_result["success"])
            results["episode_details"].append(episode_result)
            
            logger.info(f"Episode {episode + 1}/{num_episodes}: "
                       f"Reward={episode_result['total_reward']:.2f}, "
                       f"Length={episode_result['episode_length']}, "
                       f"Success={episode_result['success']}")
        
        # Calculate overall metrics
        results["success_rate"] = np.mean(results["successes"])
        results["average_reward"] = np.mean(results["episode_rewards"])
        results["average_length"] = np.mean(results["episode_lengths"])
        results["std_reward"] = np.std(results["episode_rewards"])
        results["std_length"] = np.std(results["episode_lengths"])
        
        logger.info(f"Evaluation completed. Success rate: {results['success_rate']:.3f}")
        logger.info(f"Average reward: {results['average_reward']:.2f} ± {results['std_reward']:.2f}")
        logger.info(f"Average episode length: {results['average_length']:.1f} ± {results['std_length']:.1f}")
        
        return results
    
    def _run_single_episode(self, episode_idx: int, render: bool) -> Dict[str, Any]:
        """
        Run a single evaluation episode

        Args:
            episode_idx: Episode index
            render: Whether to render the episode

        Returns:
            Dict[str, Any]: Episode results
        """
        obs, info = self.env.reset()

        # Reset model based on policy type
        if hasattr(self.model, 'reset'):
            self.model.reset()
            self.gripper_state = -1
        
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        frames = []
        
        # Get goal description for VLA agent
        if hasattr(self.env, 'obj'):
            obj_name = self.env.obj.name
        else:
            obj_name = None
        goal = self._get_goal_description(episode_idx, obj_name)
        
        episode_first_frame = True  # Track if this is the first frame for VLA agent
        while not done:
            # Update gripper state in observation for VLA agent
            obs['extra']['gripper_state'] = self.gripper_state
            
            # Get action from policy based on type
            if hasattr(self.model, 'step') and hasattr(self.model, 'base_url'):
                # VLA agent
                action = self.model.step(obs, goal, episode_first_frame=episode_first_frame)
            else:
                # Default or random policy
                action = self.model.act(obs)

            # Update gripper state (negative for environment-specific transformation)
            self.gripper_state = -action[-1]

            # Render goal site if configured
            render_goal = self.config.get("render_goal", True)
            if render_goal and hasattr(self.env, 'goal_site'):
                # Use unhide_visual instead of set_visibility for proper rendering
                # self.env.goal_site.unhide_visual()
                for v in self.env.goal_site.get_visual_bodies():
                    v.set_visibility(1.0)
            
            # Step the environment
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            # Render if requested
            if render:
                frame = self.env.render()
                if frame is not None:
                    frames.append(frame)

            episode_first_frame = False
        
        # Determine success
        success = bool(info.get("success", False))
        
        # Save video if frames were captured
        if frames and render:
            self._save_episode_video(frames, episode_idx, success)

        return {
            "episode_idx": episode_idx,
            "total_reward": episode_reward,
            "episode_length": episode_length,
            "success": success,
            "info": info
        }
    
    def _get_goal_description(self, episode_idx: int, obj_name: str) -> str:
        """
        Get goal description for VLA agent
        
        Args:
            episode_idx: Episode index
            
        Returns:
            str: Goal description
        """
        # Get goal from config or generate based on environment
        goal = self.config.get("goal", f"complete the task in episode {episode_idx}")
        if obj_name:
            obj_name = ' '.join(obj_name.split('_')[1:])
            goal = goal.format(obj_name)
        
        return goal
    
    def _save_episode_video(self, frames: List[np.ndarray], episode_idx: int, success: bool):
        """
        Save episode video
        
        Args:
            frames: List of video frames
            episode_idx: Episode index
            success: Whether the episode was successful
        """
        try:
            import cv2
            
            # Create video path
            video_dir = self.output_structure["videos_dir"]
            video_dir.mkdir(parents=True, exist_ok=True)
            
            success_str = "success" if success else "failure"
            video_path = video_dir / f"episode_{episode_idx:03d}_{success_str}.mp4"
            
            # Write video
            if frames:
                height, width = frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (width, height))
                
                for frame in frames:
                    # Convert RGB to BGR for OpenCV
                    if frame.shape[-1] == 3:
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        out.write(frame_bgr)
                
                out.release()
                logger.info(f"Video saved: {video_path}")
                
        except ImportError:
            logger.warning("OpenCV not available, skipping video saving")
        except Exception as e:
            logger.warning(f"Failed to save video: {e}")


if __name__ == "__main__":
    # Test the evaluator
    config = OmegaConf.create({
        "env_name": "PickCube-v0",
        "num_episodes": 2,
        "seed": 42,
        "policy_type": "random",
        "render": False,
        "render_mode": "cameras",
        "output_dir": "test_results"
    })
    
    # Test VLA agent configuration (commented out for now)
    # config = OmegaConf.create({
    #     "env_name": "PickCube-v0",
    #     "num_episodes": 2,
    #     "seed": 42,
    #     "policy_type": "vla",
    #     "base_url": "http://localhost:8000",
    #     "replan_step": 5,
    #     "render": False,
    #     "render_mode": "cameras",
    #     "goal": "pick up the cube",
    #     "output_dir": "test_results"
    # })
    
    output_structure = {
        "base_dir": Path("test_results"),
        "videos_dir": Path("test_results/videos"),
        "logs_dir": Path("test_results/logs")
    }
    
    evaluator = ManiSkill2Evaluator(config, output_structure)
    results = evaluator.run_evaluation()
    print(f"Test completed. Results: {results}")