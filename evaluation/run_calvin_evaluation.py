"""
Calvin Evaluation Running Script
"""

import calvin_env.envs.play_table_env as play
def _safe_get_git_commit_hash(_):
    return os.getenv("GIT_COMMIT", "unknown")
play.get_git_commit_hash = _safe_get_git_commit_hash

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
sys.path.insert(0, str(project_root))

# Import common utility functions
from evaluation.utils.tools import (
    load_config, 
    merge_config_with_args, 
    setup_logging,
    create_evaluation_output_structure,
    setup_evaluation_logging,
    save_evaluation_results,
    save_evaluation_config,
)

# Import Calvin evaluator
from evaluation.evaluator.calvin_evaluator import CalvinEvaluator

logger = logging.getLogger(__name__)

def get_calvin_default_config() -> Dict[str, Any]:
    """Get default configuration for Calvin evaluation
    
    Returns:
        Dict[str, Any]: Default configuration dictionary
    """
    return {
        # Unified output parameters
        "output_dir": "results/calvin_evaluation",
        "results_file": "results.json",
        "video_dir": "videos",
        "log_dir": "logs",
        
        # Other parameters
        "dataset_path": None,
        "debug": False,
        "device": 0,
        "temperature": 0.6,
        "replan_step": 7,
        "use_delta": True,
        "action_ensemble_horizon": 7,
        "adaptive_ensemble_alpha": 0.1,
        "action_ensemble": False
    }

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Calvin evaluation running script")
    
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
    
    args = parser.parse_args()

    return args

def main():
    """Main function"""
    args = parse_args()
    
    # Setup basic logging
    setup_logging(verbose=args.verbose)
    
    try:   
        # Load configuration
        logger.info(f"Loading configuration file: {args.config}")
        config = load_config(args.config)
        
        # Merge command line arguments into configuration
        config = merge_config_with_args(config, args)
        
        # Create output folder structure
        output_dir = config.get("output_dir", "results/calvin_evaluation")
        output_structure = create_evaluation_output_structure(
            output_dir
        )
        
        # Setup evaluation-specific logging
        setup_evaluation_logging(output_structure, verbose=args.verbose)
        
        logger.info("Starting Calvin evaluation")
        logger.info(f"Configuration: {OmegaConf.to_yaml(config)}")
        
        evaluator = CalvinEvaluator(config, output_structure)
        
        # Run evaluation
        results = evaluator.run_evaluation()
        
        # Save results and configuration
        save_evaluation_results(results, output_structure)
        save_evaluation_config(config, output_structure)
        
        logger.info("Calvin evaluation completed!")
        logger.info(f"All output files saved to: {output_structure['base_dir']}")
        return 0
        
    except Exception as e:
        logger.error(f"Error occurred during evaluation: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
