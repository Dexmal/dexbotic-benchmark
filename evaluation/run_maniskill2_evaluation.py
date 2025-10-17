#!/usr/bin/env python3
"""
ManiSkill2 Evaluation Running Script
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any
from omegaconf import OmegaConf, DictConfig

# Set project root directory path
project_root = Path(__file__).parent.parent
maniskill2_path = project_root / "maniskill2" / "ManiSkill"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(maniskill2_path))

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

# Import ManiSkill2 evaluator
from evaluation.evaluator.maniskill2_evaluator import ManiSkill2Evaluator

logger = logging.getLogger(__name__)


def get_maniskill2_default_config() -> Dict[str, Any]:
    """
    Get default configuration for ManiSkill2 evaluation
    
    Returns:
        Dict[str, Any]: Default configuration dictionary
    """
    return {
        # Environment configuration
        "env_name": "PickCube-v0",
        
        # Evaluation parameters
        "num_episodes": 10,
        "seed": 42,
        
        # Policy configuration
        "policy_type": "random",
        
        # Rendering and visualization
        "render": True,
        
        # Unified output parameters
        "output_dir": "results/maniskill2_evaluation",
        "results_file": "results.json",
        "video_dir": "videos",
        "log_dir": "logs",
        
        # Other parameters
        "log_level": "INFO"
    }


def parse_args():
    """
    Parse command line arguments    
    """
    parser = argparse.ArgumentParser(
        description="ManiSkill2 environment evaluation running script",
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
    required_keys = ["env_name"]
    missing_keys = [key for key in required_keys if key not in config or config[key] is None]
    
    if missing_keys:
        raise ValueError(f"Missing required configuration items: {missing_keys}")
    
    # Validate supported environment names
    supported_envs = ["PickCube-v0", "StackCube-v0", "PickSingleYCB-v0", 
                     "PickSingleEGAD-v0", "PickClutterYCB-v0"]
    if config.env_name not in supported_envs:
        raise ValueError(f"Unsupported environment: {config.env_name}. Supported environments: {supported_envs}")


def main():
    """
    Main function
    
    Execute complete ManiSkill2 evaluation process
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
            default_values = get_maniskill2_default_config()
            config = create_default_config(default_values)
        
        # Set task name from command line argument
        
        # Merge command line arguments into configuration
        config = merge_config_with_args(config, args)
        
        # Validate required configuration
        validate_required_config(config)
       
        # Create output folder structure
        output_dir = config.get("output_dir", "results/maniskill2_evaluation")
        # Include task name in output directory
        task_name = f"maniskill2_{config.env_name.replace('-v0', '')}"
        output_structure = create_evaluation_output_structure(
            output_dir, 
            task_name=task_name
        )

        # Setup evaluation-specific logging
        setup_evaluation_logging(output_structure, verbose=args.verbose)

        logger.info("Starting ManiSkill2 evaluation")
        logger.info(f"Configuration: {OmegaConf.to_yaml(config)}")
        
        # Create evaluator
        logger.info("Creating evaluator...")
        evaluator = ManiSkill2Evaluator(config, output_structure)
        
        # Run evaluation
        logger.info("Starting evaluation...")
        results = evaluator.run_evaluation()
        
        # Save results and configuration
        save_evaluation_results(results, output_structure)
        save_evaluation_config(config, output_structure)
        
        logger.info("ManiSkill2 evaluation completed!")
        logger.info(f"All output files saved to: {output_structure['base_dir']}")
        return 0
        
    except Exception as e:
        logger.error(f"Error occurred during evaluation: {e}")
        import traceback
        logger.error(f"Detailed error information: {traceback.format_exc()}")
        return 1

# import debugpy
# try:
#     debugpy.listen(("localhost", 2346))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     raise e

if __name__ == "__main__":
    exit(main())