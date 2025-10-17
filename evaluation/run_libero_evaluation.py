"""
Libero Evaluation Running Script
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any
from omegaconf import OmegaConf, DictConfig

project_root = Path(__file__).parent.parent
libero_path = project_root / "libero" / "libero" / "libero"
os.environ["LIBERO_CONFIG_PATH"] = str(libero_path)
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


from evaluation.evaluator.libero_evaluator import LiberoEvaluator

logger = logging.getLogger(__name__)


def get_libero_default_config() -> Dict[str, Any]:
    """Get default configuration for Libero evaluation
    
    Returns:
        Dict[str, Any]: Default configuration dictionary
    """
    return {
        # Unified output parameters
        "output_dir": "results/libero_evaluation",
        "results_file": "results.json",
        "video_dir": "videos",
        "log_dir": "logs",
        
        # Evaluation parameters
        "benchmark": "libero_spatial",
        "num_trails_per_task": 1,
        "num_steps_wait": 10,
        "seed": 42,
        
        # Other parameters
        "log_level": "INFO"
    }


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Libero evaluation running script")
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

def main():
    """Main function"""
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
            default_values = get_libero_default_config()
            config = create_default_config(default_values)
        
        # Merge command line arguments into configuration
        config = merge_config_with_args(config, args)
        
        # Create output folder structure
        output_dir = config.get("output_dir", "results/libero_evaluation")
        output_structure = create_evaluation_output_structure(
            output_dir
        )
        
        # Setup evaluation-specific logging
        setup_evaluation_logging(output_structure, verbose=args.verbose)
        
        logger.info("Starting Libero evaluation")
        logger.info(f"Configuration: {OmegaConf.to_yaml(config)}")
        
        evaluator = LiberoEvaluator(config, output_structure)
        
        # Run evaluation
        results = evaluator.run_evaluation()
        
        # Save results and configuration
        save_evaluation_results(results, output_structure)
        save_evaluation_config(config, output_structure)
        
        logger.info("Libero evaluation completed!")
        logger.info(f"All output files saved to: {output_structure['base_dir']}")
        return 0
        
    except Exception as e:
        logger.error(f"Error occurred during evaluation: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
