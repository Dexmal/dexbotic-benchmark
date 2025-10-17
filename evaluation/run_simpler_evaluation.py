#!/usr/bin/env python3
"""
Simpler Evaluation Running Script
"""

import argparse
import json
import logging
import os
import sys
import shutil
from pathlib import Path
from typing import Dict, Any
from omegaconf import OmegaConf, DictConfig

# Set project root directory path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import common utility functions
from evaluation.utils.tools import (
    load_config, 
    merge_config_with_args, 
    create_default_config, 
    setup_logging,
    create_evaluation_output_structure,
    setup_evaluation_logging,
    save_evaluation_results,
    save_evaluation_config
)

# Import Simpler evaluator
from evaluation.evaluator.simpler_evaluator import SimplerEvaluator

logger = logging.getLogger(__name__)


def get_simpler_default_config() -> Dict[str, Any]:
    """
    Get default configuration for Simpler evaluation
    
    Returns:
        Dict[str, Any]: Default configuration dictionary
    """
    return {
        # Policy model configuration
        "policy_model": "vla",
        "policy_setup": "google_robot",
        "ckpt_path": None,
        
        # Environment configuration
        "env_name": "StackGreenCubeOnYellowCubeBakedTexInScene-v0",
        "scene_name": "google_pick_coke_can_1_v4",
        "robot": "google_robot_static",
        "enable_raytracing": False,
        
        # Observation and action parameters
        "obs_camera_name": None,
        "action_scale": 1.0,
        
        # Control parameters
        "control_freq": 3,
        "sim_freq": 513,
        "max_episode_steps": 80,
        "rgb_overlay_path": None,
        
        # Robot initial position parameters
        "robot_init_x_range": [0.35, 0.35, 1],
        "robot_init_y_range": [0.20, 0.20, 1],
        "robot_init_rot_quat_center": [1, 0, 0, 0],
        "robot_init_rot_rpy_range": [0, 0, 1, 0, 0, 1, 0, 0, 1],
        
        # Object variation parameters
        "obj_variation_mode": "xy",
        "obj_episode_range": [0, 60],
        "obj_init_x_range": [-0.35, -0.12, 5],
        "obj_init_y_range": [-0.02, 0.42, 5],
        
        # Other environment parameters
        "additional_env_save_tags": None,
        "additional_env_build_kwargs": {},
        
        # Unified output parameters
        "output_dir": "results/simpler_evaluation",
        "results_file": "results.json",
        "video_dir": "videos",
        "log_dir": "logs",
        
        # Other parameters
        "log_level": "INFO",
        
        # System parameters
        "tf_memory_limit": 3072,
        "octo_init_rng": 0,
    }


def parse_args():
    """
    Parse command line arguments    
    """
    parser = argparse.ArgumentParser(
        description="Simpler ManiSkill2 environment evaluation running script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration file parameters
    parser.add_argument(
        "--config", 
        type=str, 
        help="Configuration file path (YAML format)"
    )
    
    # System parameters
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    # Add a general parameter parser that supports arbitrary key-value pairs
    parser.add_argument(
        "--set",
        nargs=2,
        metavar=('KEY', 'VALUE'),
        action='append',
        help="Set configuration parameters, format: --set key value. Can be used multiple times to set multiple parameters"
    )
    
    return parser.parse_args()

def validate_required_config(config: DictConfig) -> None:
    """
    Validate required configuration parameters
    
    Args:
        config: Configuration object
        
    Raises:
        ValueError: When required configuration is missing
    """
    required_keys = ["policy_model", "env_name", "robot"]
    missing_keys = [key for key in required_keys if key not in config or config[key] is None]
    
    if missing_keys:
        raise ValueError(f"Missing required configuration items: {missing_keys}")
    
    # Validate required parameters for specific models
    if config.policy_model == "rt1" and not config.get("ckpt_path"):
        raise ValueError("RT1 model requires specifying checkpoint path (ckpt_path)")


def move_ckpt_folders_to_video_dir(logs_dir: Path, videos_dir: Path) -> None:
    """
    Move ckpt_name folders under logging_dir to video directory
    
    Args:
        logs_dir: Log directory path
        videos_dir: Video directory path
    """
    logger.info("Starting to move ckpt folders to video directory...")
    
    # Ensure video directory exists
    videos_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all folders under logs directory (ckpt_name)
    ckpt_dirs = [item for item in logs_dir.iterdir() if item.is_dir()]
    
    if not ckpt_dirs:
        logger.info("No ckpt folders found")
        return
    
    logger.info(f"Found {len(ckpt_dirs)} ckpt folders")
    
    # Move ckpt folders
    moved_count = 0
    for ckpt_dir in ckpt_dirs:
        try:
            ckpt_name = ckpt_dir.name
            target_dir = videos_dir / ckpt_name
            
            # If target directory already exists, add sequence number
            counter = 1
            while target_dir.exists():
                target_dir = videos_dir / f"{ckpt_name}_{counter}"
                counter += 1
            
            # Move entire folder
            shutil.move(str(ckpt_dir), str(target_dir))
            moved_count += 1
            
            logger.info(f"Moved ckpt folder: {ckpt_name} -> {target_dir.name}")
                
        except Exception as e:
            logger.warning(f"Error moving ckpt folder {ckpt_dir}: {e}")
    
    logger.info(f"Successfully moved {moved_count} ckpt folders to {videos_dir}")


def cleanup_empty_dirs(directory: Path) -> None:
    """
    Clean up empty directories
    
    Args:
        directory: Directory to clean up
    """
    try:
        # Recursively delete empty directories
        for item in directory.rglob("*"):
            if item.is_dir() and not any(item.iterdir()):
                item.rmdir()
                logger.debug(f"Deleted empty directory: {item}")
    except Exception as e:
        logger.warning(f"Error cleaning up empty directories: {e}")


def main():
    """
    Main function
    
    Execute complete Simpler evaluation process
    """
    args = parse_args()
    
    # Setup basic logging
    setup_logging(verbose=args.verbose)
    
    try:   
        # Load configuration
        if args.config:
            logger.info(f"Loading configuration file: {args.config}")
            config = load_config(args.config)
        else:
            logger.info("Using default configuration")
            default_values = get_simpler_default_config()
            config = create_default_config(default_values)
        
        # Merge command line arguments into configuration
        config = merge_config_with_args(config, args)
        
        # Validate required configuration
        validate_required_config(config)
       
        # Create output folder structure
        output_dir = config.get("output_dir", "results/simpler_evaluation")
        output_structure = create_evaluation_output_structure(
            output_dir, 
            task_name="simpler_evaluation"
        )

        # Setup evaluation-specific logging
        setup_evaluation_logging(output_structure, verbose=args.verbose)

        logger.info("Starting Simpler evaluation")
        logger.info(f"Configuration: {OmegaConf.to_yaml(config)}")
        
        # Create evaluator
        logger.info("Creating evaluator...")
        evaluator = SimplerEvaluator(config, output_structure)
        
        # Run evaluation
        logger.info("Starting evaluation...")
        results = evaluator.run_evaluation()
        
        # Save results and configuration
        save_evaluation_results(results, output_structure)
        save_evaluation_config(config, output_structure)
        
        # Post-processing: move video files
        logger.info("Starting post-processing: moving video files...")
        move_ckpt_folders_to_video_dir(
            logs_dir=output_structure["logs_dir"],
            videos_dir=output_structure["videos_dir"]
        )
        
        logger.info("Simpler evaluation completed!")
        logger.info(f"All output files saved to: {output_structure['base_dir']}")
        return 0
        
    except MemoryError as e:
        logger.error(f"Memory insufficient error: {e}")
        logger.error("Suggestion: reduce tf_memory_limit or use smaller model")
        return 1
    except Exception as e:
        logger.error(f"Error occurred during evaluation: {e}")
        import traceback
        logger.error(f"Detailed error information: {traceback.format_exc()}")
        return 1
    finally:
        # Clean up memory
        cleanup_memory()


def cleanup_memory():
    """
    Clean up memory
    """
    import gc
    
    # Clean up Python garbage collection
    gc.collect()
    
    # Clean up PyTorch cache
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    
    logger.info("Memory cleanup completed")


if __name__ == "__main__":
    main()
