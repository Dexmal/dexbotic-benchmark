"""
Libero Environment Evaluator
"""

from pathlib import Path
from typing import Dict, Any, List
import logging
import math
import time
import numpy as np
import tqdm
import imageio
import os
import sys
from PIL import Image

from .base_evaluator import BaseEvaluator

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

# Import EGL device setup functions
try:
    from calvin.calvin_env.calvin_env.utils.utils import set_egl_device
    EGL_UTILS_AVAILABLE = True
except ImportError:
    EGL_UTILS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("EGL utils not available, EGL device setup will be skipped")

logger = logging.getLogger(__name__)

# Libero constants
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

MAX_STEP_MAPPING = {
    'libero_spatial': 220,
    'libero_goal': 300,
    'libero_object': 280,
    'libero_10': 520,
    'libero_90': 400,
}


def convert_to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert image to uint8 format if it is floating point"""
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    return img


def resize_with_pad(images: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    """Use PIL to replicate tf.image.resize_with_pad functionality"""
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape
    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([_resize_with_pad_pil(Image.fromarray(im), height, width, method=method) for im in images])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _resize_with_pad_pil(image: Image.Image, height: int, width: int, method: int) -> Image.Image:
    """Use PIL to replicate tf.image.resize_with_pad functionality"""
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    return zero_image


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


class LiberoEvaluator(BaseEvaluator):
    """
    Libero Environment Evaluator
    """
    def __init__(self, config: Dict[str, Any], output_structure: Dict[str, str]):
        """
        Initialize Libero evaluator
        
        Args:
            config: Configuration dictionary
            output_structure: Output directory structure
        """
        super().__init__(config, output_structure)
        self._setup_egl_environment()
    
    def _setup_egl_environment(self) -> None:
        """
        Setup EGL environment to avoid rendering errors
        
        This method sets necessary environment variables and EGL device configuration
        to resolve EGL context errors in headless server environments
        """
        try:
            # Set EGL-related environment variables
            os.environ.setdefault("EGL_PLATFORM", "device")
            os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
            
            # If EGL utils are available, setup EGL device
            if EGL_UTILS_AVAILABLE:
                try:
                    # Try to setup EGL device (using GPU 0)
                    import torch
                    if torch.cuda.is_available():
                        device = torch.device("cuda:0")
                        set_egl_device(device)
                        logger.info("EGL device setup completed successfully")
                    else:
                        logger.warning("CUDA not available, using default EGL device")
                except Exception as e:
                    logger.warning(f"Failed to setup EGL device: {e}")
                    # Set default EGL device
                    os.environ.setdefault("EGL_VISIBLE_DEVICES", "0")
            else:
                # If EGL utils are not available, set default values
                os.environ.setdefault("EGL_VISIBLE_DEVICES", "0")
                logger.info("Using default EGL device configuration")
                
        except Exception as e:
            logger.error(f"Error setting up EGL environment: {e}")
            # Set basic EGL environment variables as fallback
            os.environ.setdefault("EGL_VISIBLE_DEVICES", "0")
            os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


    def setup_environment(self) -> Any:
        """Setup Libero environment"""
        return None
    
    def setup_model(self) -> Any:
        """Setup Libero model"""
        try:
            from evaluation.policies.libero_vla_agent import LiberoVLAAgent
            
            logger.info("Loading Libero model")
            model = LiberoVLAAgent(self.config)
            
            return model
            
        except ImportError as e:
            logger.error(f"Unable to import Libero model: {e}")
            raise
    
    def _run_evaluation_impl(self) -> Dict[str, Any]:
        """Implement Libero evaluation logic"""
        try:
            # Set random seed
            np.random.seed(self.config.seed)
            # Get benchmark and task suite
            benchmark_dict = benchmark.get_benchmark_dict()
            task_suite = benchmark_dict[self.config.benchmark]()
            num_tasks_in_suite = task_suite.n_tasks
            num_trails_per_task = self.config.num_trails_per_task
            num_steps_wait = self.config.num_steps_wait
            video_out_path = self.output_structure['videos_dir']
            
            logging.info(f"Evaluating {num_tasks_in_suite} tasks in {self.config.benchmark} benchmark")

            # Ensure video output directory exists
            Path(video_out_path).mkdir(parents=True, exist_ok=True)
            
            max_step = MAX_STEP_MAPPING[self.config.benchmark]
            
            # Start evaluation
            total_episodes, total_successes = 0, 0
            all_task_results = []
            
            for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
                # Get task
                task = task_suite.get_task(task_id)
                task_description = task.language
                task_id_str = getattr(task, 'id', f'task_{task_id}')
                
                # Get default LIBERO initial states
                initial_states = task_suite.get_task_init_states(task_id)
  
                # Initialize LIBERO environment and task description
                self.env, _ = self._get_libero_env(task, LIBERO_ENV_RESOLUTION, self.config.seed)
                
                # Start episode
                task_episodes, task_successes = 0, 0
                task_results = []
                
                for episode_idx in tqdm.tqdm(range(num_trails_per_task)):
                    episode_first_frame = True
                    episode_start_time = time.time()
                    steps = 0
                    success = False
                    replay_images = []
                    
                    try:
                        logger.info(f"\nTask: {task_description}")
                        
                        # Reset environment
                        self.env.reset()
                        self.model.reset()
                        
                        # Set initial state
                        obs = self.env.set_init_state(initial_states[episode_idx])
        
                        logger.info(f"Starting episode {task_episodes+1}...")
                        
                        while steps < max_step + num_steps_wait:
                            try:
                                # Do nothing in the first few timesteps, wait for objects to fall
                                if steps < num_steps_wait:
                                    obs, reward, done, info = self.env.step(LIBERO_DUMMY_ACTION)
                                    steps += 1
                                    continue
                                
                                # Get preprocessed images
                                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                                
                                # Image preprocessing
                                img = convert_to_uint8(
                                    resize_with_pad(img, LIBERO_ENV_RESOLUTION, LIBERO_ENV_RESOLUTION)
                                )
                                wrist_img = convert_to_uint8(
                                    resize_with_pad(wrist_img, LIBERO_ENV_RESOLUTION, LIBERO_ENV_RESOLUTION)
                                )
                                
                                # Save preprocessed images for replay video
                                replay_images.append(img)
                                observation = dict()
                                goal = task_description

                                if self.config.get('send_state', False):
                                    observation['state'] = np.concatenate([
                                        obs['robot0_eef_pos'],
                                        _quat2axisangle(obs['robot0_eef_quat']),
                                        obs['robot0_gripper_qpos']
                                    ])

                                images = []
                                if self.config.get('send_image', []):
                                    if 'image' in self.config['send_image']:
                                        images.append(img)
                                    if 'wrist_image' in self.config['send_image']:
                                        images.append(wrist_img)
                                if not images:
                                    images.append(img)
                                observation['image'] = images
                                
                                # Use step method (standard method for LiberoVLAAgent)
                                action = self.model.step(observation, goal, episode_first_frame=episode_first_frame)
                                
                                # Execute action in environment
                                obs, reward, done, info = self.env.step(action)
                                if done:
                                    success = True
                                    task_successes += 1
                                    total_successes += 1
                                    break
                                steps += 1
                                episode_first_frame = False
                                
                            except Exception as e:
                                logger.error(f"Caught exception: {e}")
                                break
                        
                        episode_time = time.time() - episode_start_time
                        
                        # Save episode replay video
                        suffix = "success" if success else "failure"
                        task_segment = task_description.replace(" ", "_")
                        video_path = Path(video_out_path) / f"rollout_{task_segment}_{episode_idx}_{suffix}.mp4"
                        
                        try:
                            imageio.mimwrite(
                                video_path,
                                [np.asarray(x) for x in replay_images],
                                fps=10,
                            )
                        except Exception as e:
                            logger.warning(f"Failed to save video: {e}")
                        
                        episode_result = {
                            "episode": episode_idx,
                            "success": success,
                            "steps": steps,
                            "time": episode_time,
                            "video_path": str(video_path)
                        }
                        
                        task_results.append(episode_result)
                        task_episodes += 1
                        total_episodes += 1
                        
                        logger.info(f"Success: {success}")
                        logger.info(f"# episodes completed so far: {total_episodes}")
                        logger.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
                        
                    except Exception as e:
                        logger.error(f"Episode {episode_idx} execution failed: {e}")
                        task_results.append({
                            "episode": episode_idx,
                            "success": False,
                            "steps": 0,
                            "time": time.time() - episode_start_time,
                            "error": str(e)
                        })
                        task_episodes += 1
                        total_episodes += 1
                
                # Record task results
                task_success_rate = task_successes / task_episodes if task_episodes > 0 else 0.0
                logger.info(f"Current task success rate: {task_success_rate:.2%}")
                logger.info(f"Current total success rate: {total_successes / total_episodes:.2%}")
                
                task_summary = {
                    "task_id": task_id_str,
                    "task_description": task_description,
                    "total_episodes": task_episodes,
                    "successful_episodes": task_successes,
                    "success_rate": task_success_rate,
                    "episode_results": task_results
                }
                
                all_task_results.append(task_summary)
            
            # Calculate overall results
            overall_success_rate = total_successes / total_episodes if total_episodes > 0 else 0.0
            
            result = {
                "total_tasks": num_tasks_in_suite,
                "total_episodes": total_episodes,
                "successful_episodes": total_successes,
                "success_rate": overall_success_rate,
                "task_results": all_task_results
            }
            
            logger.info(f"Total success rate: {overall_success_rate:.2%}")
            logger.info(f"Total episodes: {total_episodes}")
            
            # Cleanup EGL context
            self._cleanup_egl_context()
            
            return result
            
        except Exception as e:
            logger.error(f"Error occurred during Libero evaluation: {e}")
            # Ensure EGL context cleanup even in exception cases
            self._cleanup_egl_context()
            return {
                "error": str(e),
                "total_tasks": 0,
                "total_episodes": 0,
                "successful_episodes": 0,
                "success_rate": 0.0,
                "task_results": []
            }
    
    def _get_libero_env(self, task, resolution, seed):
        """
        Initialize and return LIBERO environment, also return task description
        
        Args:
            task: LIBERO task object
            resolution (int): Environment resolution
            seed (int): Random seed
            
        Returns:
            tuple: (environment object, task description)
        """

        task_description = task.language
        task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
        env = OffScreenRenderEnv(**env_args)
        env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
        return env, task_description

    def _cleanup_egl_context(self) -> None:
        """
        Cleanup EGL context to avoid memory leaks and errors
        
        This method is called at the end of evaluation to ensure EGL context is properly cleaned up
        """
        try:
            # If environment object exists, try to cleanup its rendering context
            if hasattr(self, 'env') and self.env is not None:
                try:
                    # Try to close environment renderer
                    if hasattr(self.env, 'close'):
                        self.env.close()
                    elif hasattr(self.env, 'renderer') and hasattr(self.env.renderer, 'close'):
                        self.env.renderer.close()
                except Exception as e:
                    logger.warning(f"Error closing environment renderer: {e}")
            
            # Cleanup OpenGL context
            try:
                import OpenGL
                if hasattr(OpenGL, 'EGL'):
                    # Try to cleanup EGL context
                    pass  # EGL context usually cleans up automatically
            except ImportError:
                pass
                
            logger.info("EGL context cleanup completed")
            
        except Exception as e:
            logger.warning(f"Error during EGL context cleanup: {e}")
    