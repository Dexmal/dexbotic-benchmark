"""
Evaluate a model on ManiSkill2 environment.
"""

import os
import logging
import time
import warnings
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import tqdm
from transforms3d.euler import quat2euler

from simpler_env.utils.env.env_builder import build_maniskill2_env, get_robot_control_mode
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from simpler_env.utils.visualization import write_video

# Suppress all deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*deprecated.*")
warnings.filterwarnings("ignore", message=".*robot_uid.*")
warnings.filterwarnings("ignore", message=".*get_language_instruction.*")
warnings.filterwarnings("ignore", message=".*is_final_subtask.*")

# Setup logger
logger = logging.getLogger(__name__)


def run_maniskill2_eval_single_episode(
    model: Any,
    robot_name: str,
    env_name: str,
    scene_name: str,
    robot_init_x: float,
    robot_init_y: float,
    robot_init_quat: List[float],
    control_mode: str,
    obj_init_x: Optional[float] = None,
    obj_init_y: Optional[float] = None,
    obj_episode_id: Optional[int] = None,
    additional_env_build_kwargs: Optional[Dict[str, Any]] = None,
    rgb_overlay_path: Optional[str] = None,
    obs_camera_name: Optional[str] = None,
    control_freq: int = 3,
    sim_freq: int = 513,
    max_episode_steps: int = 20,  # Reduced for quick testing
    instruction: Optional[str] = None,
    enable_raytracing: bool = False,
    additional_env_save_tags: Optional[str] = None,
    logging_dir: str = "./results",
    video_path: Optional[str] = None
) -> bool:
    """
    Run a single episode evaluation in ManiSkill2 environment
    
    This function is responsible for creating environment, initializing model, 
    executing tasks and recording results. Supports multiple robot initial 
    positions and object variation modes.
    
    Args:
        model: Policy model to evaluate
        robot_name: Robot name
        env_name: Environment name
        scene_name: Scene name
        robot_init_x: Robot initial x coordinate
        robot_init_y: Robot initial y coordinate
        robot_init_quat: Robot initial quaternion rotation
        control_mode: Control mode
        obj_init_x: Object initial x coordinate (optional)
        obj_init_y: Object initial y coordinate (optional)
        obj_episode_id: Object episode ID (optional)
        additional_env_build_kwargs: Additional environment build parameters
        rgb_overlay_path: RGB overlay image path
        obs_camera_name: Observation camera name
        control_freq: Control frequency
        sim_freq: Simulation frequency
        max_episode_steps: Maximum episode steps
        instruction: Task instruction
        enable_raytracing: Whether to enable ray tracing
        additional_env_save_tags: Additional save tags
        logging_dir: Logging directory
        video_path: Video save path
        
    Returns:
        bool: Whether the task was completed successfully
    """

    # Initialize parameters
    if additional_env_build_kwargs is None:
        additional_env_build_kwargs = {}
    
    episode_start_time = time.time()

    # Create environment configuration
    kwargs = dict(
        obs_mode="rgbd",
        robot=robot_name,
        sim_freq=sim_freq,
        control_mode=control_mode,
        control_freq=control_freq,
        max_episode_steps=max_episode_steps,
        scene_name=scene_name,
        camera_cfgs={"add_segmentation": True},
        rgb_overlay_path=rgb_overlay_path,
    )
    
    # Handle ray tracing configuration
    if enable_raytracing:
        ray_tracing_dict = {"shader_dir": "rt"}
        ray_tracing_dict.update(additional_env_build_kwargs)
        # Put ray tracing dict keys before other keys for compatibility with existing result naming and metric calculation
        additional_env_build_kwargs = ray_tracing_dict
    
    # Build ManiSkill2 environment
    env = build_maniskill2_env(
        env_name,
        **additional_env_build_kwargs,
        **kwargs,
    )

    env_reset_options = {
        "robot_init_options": {
            "init_xy": np.array([robot_init_x, robot_init_y]),
            "init_rot_quat": robot_init_quat,
        }
    }
    
    # Set object variation mode
    if obj_init_x is not None:
        assert obj_init_y is not None, "obj_init_y must be specified when obj_init_x is provided"
        obj_variation_mode = "xy"
        env_reset_options["obj_init_options"] = {
            "init_xy": np.array([obj_init_x, obj_init_y]),
        }
    else:
        assert obj_episode_id is not None, "obj_episode_id must be specified when obj_init_x is not provided"
        obj_variation_mode = "episode"
        env_reset_options["obj_init_options"] = {
            "episode_id": obj_episode_id,
        }
    
    # Reset environment
    obs, _ = env.reset(options=env_reset_options)
    
    # Check if current subtask is the final subtask (for long-horizon environments)
    try:
        is_final_subtask = env.unwrapped.is_final_subtask()
    except AttributeError:
        # Fallback to wrapper method if unwrapped doesn't have the attribute
        is_final_subtask = env.get_wrapper_attr('is_final_subtask')() 

    # Get language instruction
    if instruction is not None:
        task_description = instruction
    else:
        # Get default language instruction
        try:
            task_description = env.unwrapped.get_language_instruction()
        except AttributeError:
            # Fallback to wrapper method if unwrapped doesn't have the attribute
            task_description = env.get_wrapper_attr('get_language_instruction')()

    # Initialize recording variables
    image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
    images = [image]
    predicted_actions = []
    predicted_terminated, done, truncated = False, False, False

    # Initialize model
    model.reset(task_description)

    timestep = 0
    success = "failure"
    episode_first_frame = True

    # Execute environment steps
    while not (predicted_terminated or truncated):
        try:
            # Model inference step
            # "raw_action" is raw model action output; "action" is processed action to be sent to maniskill env
            raw_action, action = model.step(image, task_description, episode_first_frame=episode_first_frame)
            episode_first_frame = False
            predicted_actions.append(raw_action)
            predicted_terminated = bool(action["terminate_episode"][0] > 0)
            
            # Handle subtask switching
            if predicted_terminated:
                if not is_final_subtask:
                    # Advance environment to next subtask
                    predicted_terminated = False
                    env.advance_to_next_subtask()

            # Execute environment step
            obs, reward, done, truncated, info = env.step(
                np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]]),
            )
            
            # Update success status
            success = "success" if done else "failure"
            
            # Check if task description has changed
            try:
                new_task_description = env.unwrapped.get_language_instruction()
            except AttributeError:
                # Fallback to wrapper method if unwrapped doesn't have the attribute
                new_task_description = env.get_wrapper_attr('get_language_instruction')()
            
            if new_task_description != task_description:
                task_description = new_task_description
            
            try:
                is_final_subtask = env.unwrapped.is_final_subtask()
            except AttributeError:
                # Fallback to wrapper method if unwrapped doesn't have the attribute
                is_final_subtask = env.get_wrapper_attr('is_final_subtask')()

            # Get new image
            image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
            images.append(image)
            timestep += 1
                
        except Exception as e:
            logger.error(f"Step {timestep} execution failed: {e}")
            break

    # Get episode statistics
    episode_stats = info.get("episode_stats", {})
    episode_time = time.time() - episode_start_time

    env_save_name = env_name
    for k, v in additional_env_build_kwargs.items():
        env_save_name = env_save_name + f"_{k}_{v}"
    if additional_env_save_tags is not None:
        env_save_name = env_save_name + f"_{additional_env_save_tags}"
    
    # Generate video name based on object variation mode
    if obj_variation_mode == "xy":
        video_name = f"{success}_obj_{obj_init_x}_{obj_init_y}"
    elif obj_variation_mode == "episode":
        video_name = f"{success}_obj_episode_{obj_episode_id}"
    
    # Add episode statistics to video name
    for k, v in episode_stats.items():
        video_name = video_name + f"_{k}_{v}"
    video_name = video_name + ".mp4"
    
    # Handle RGB overlay path
    if rgb_overlay_path is not None:
        rgb_overlay_path_str = os.path.splitext(os.path.basename(rgb_overlay_path))[0]
    else:
        rgb_overlay_path_str = "None"
    
    # Generate video save path
    r, p, y = quat2euler(robot_init_quat)
    video_path = f"{video_path}/{scene_name}/{control_mode}/{env_save_name}/rob_{robot_init_x}_{robot_init_y}_rot_{r:.3f}_{p:.3f}_{y:.3f}_rgb_overlay_{rgb_overlay_path_str}/{video_name}"
    
    # Ensure directory exists and save video
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    write_video(video_path, images, fps=5)
    # logger.info(f"Video saved to: {video_path}")

    # Save action trajectory
    # logger.info("Saving action trajectory...")
    action_path = video_path.replace(".mp4", ".png")
    action_root = os.path.dirname(action_path) + "/actions/"
    os.makedirs(action_root, exist_ok=True)
    action_path = action_root + os.path.basename(action_path)
    model.visualize_epoch(predicted_actions, images, save_path=action_path)
    # logger.info(f"Action trajectory saved to: {action_path}")

    return success == "success"


def maniskill2_evaluator(model: Any, args: Any) -> List[bool]:
    """
    Main ManiSkill2 environment evaluator function
    
    This function coordinates the entire evaluation process, including iterating through 
    all robot initial positions and object variation configurations, executing multiple 
    episode evaluations, and collecting success rate statistics.
    
    Args:
        model: Policy model to evaluate
        args: Evaluation arguments object containing all necessary configuration information
        
    Returns:
        List[bool]: List of success status for each episode
        
    Raises:
        NotImplementedError: Unsupported object variation mode
    """
    # Get robot control mode
    control_mode = get_robot_control_mode(args.robot, args.policy_model)
    success_arr = []
    
    # Calculate total episode count for full evaluation
    total_robot_configs = len(args.robot_init_xs) * len(args.robot_init_ys) * len(args.robot_init_quats)
    
    if args.obj_variation_mode == "xy":
        total_obj_configs = len(args.obj_init_xs) * len(args.obj_init_ys)
    elif args.obj_variation_mode == "episode":
        total_obj_configs = args.obj_episode_range[1] - args.obj_episode_range[0]
    else:
        raise NotImplementedError(f"Unsupported object variation mode: {args.obj_variation_mode}")
    
    total_episodes = total_robot_configs * total_obj_configs
    
    logger.info(f"Starting ManiSkill2 evaluation")
    logger.info(f"Environment: {args.env_name}, Scene: {args.scene_name}, Robot: {args.robot}")
    logger.info(f"Robot configs: {total_robot_configs}, Object configs: {total_obj_configs}")
    logger.info(f"Total episodes: {total_episodes}")
    
    episode_count = 0
    successful_episodes = 0
    
    # Create progress bar with libero-style format
    pbar = tqdm.tqdm(total=total_episodes, desc="", unit="episode", 
                     bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    # Execute inference evaluation with full configuration
    for robot_init_x in args.robot_init_xs:
        for robot_init_y in args.robot_init_ys:
            for robot_init_quat in args.robot_init_quats:
                # Prepare basic parameters
                kwargs = dict(
                    model=model,
                    robot_name=args.robot,
                    env_name=args.env_name,
                    scene_name=args.scene_name,
                    robot_init_x=robot_init_x,
                    robot_init_y=robot_init_y,
                    robot_init_quat=robot_init_quat,
                    control_mode=control_mode,
                    additional_env_build_kwargs=args.additional_env_build_kwargs,
                    rgb_overlay_path=args.rgb_overlay_path,
                    control_freq=args.control_freq,
                    sim_freq=args.sim_freq,
                    max_episode_steps=args.max_episode_steps,
                    enable_raytracing=args.enable_raytracing,
                    additional_env_save_tags=args.additional_env_save_tags,
                    obs_camera_name=args.obs_camera_name,
                    logging_dir=args.logging_dir,
                    video_path=args.video_path,
                )
                
                # Execute evaluation based on object variation mode
                if args.obj_variation_mode == "xy":
                    for obj_init_x in args.obj_init_xs:
                        for obj_init_y in args.obj_init_ys:
                            episode_count += 1
                            
                            success = run_maniskill2_eval_single_episode(
                                obj_init_x=obj_init_x,
                                obj_init_y=obj_init_y,
                                **kwargs,
                            )
                            success_arr.append(success)
                            
                            if success:
                                successful_episodes += 1
                            
                            # Update progress bar
                            pbar.update(1)
                            
                            # Log current statistics in libero style
                            current_success_rate = successful_episodes / episode_count
                            logger.info(f"# successes: {successful_episodes} ({current_success_rate:.1%})")
                            
                elif args.obj_variation_mode == "episode":
                    for obj_episode_id in range(args.obj_episode_range[0], args.obj_episode_range[1]):
                        episode_count += 1
                        
                        success = run_maniskill2_eval_single_episode(
                            obj_episode_id=obj_episode_id, 
                            **kwargs
                        )
                        success_arr.append(success)
                        
                        if success:
                            successful_episodes += 1
                        
                        # Update progress bar
                        pbar.update(1)
                        
                        # Log current statistics in libero style
                        current_success_rate = successful_episodes / episode_count
                        logger.info(f"# successes: {successful_episodes} ({current_success_rate:.1%})")
    
    # Close progress bar
    pbar.close()

    return success_arr
