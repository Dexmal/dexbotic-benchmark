#!/usr/bin/env python3
"""
Simpler Results Aggregation Script

This script aggregates success rates for environments with the same name, 
calculates averages and generates summary reports.
Supports reading result.json files from multiple result directories, 
grouping by environment name and calculating statistics.

Usage:
python scripts/aggregate_simpler_results.py --base-dir results/ --output-file aggregated_results.json

Features:
- Recursively search for all result.json files in specified directories
- Group results by environment name (env_name)
- Calculate average success rates, standard deviations and other statistics for each environment
- Generate detailed summary reports
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import statistics

# Set project root directory path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True  # Force reconfiguration of logging
)
logger = logging.getLogger(__name__)


def find_result_files(base_dir: Path) -> List[Path]:
    """
    Recursively find all result.json files in the specified directory
    
    Args:
        base_dir: Base search directory
        
    Returns:
        List[Path]: List of found result.json file paths
    """
    result_files = []
    
    if not base_dir.exists():
        logger.error(f"Base directory does not exist: {base_dir}")
        return result_files
    
    # Recursively search for all result.json files
    for result_file in base_dir.rglob("results.json"):
        if result_file.is_file():
            result_files.append(result_file)
            logger.debug(f"Found result file: {result_file}")
            
    return result_files


def load_result_file(file_path: Path) -> Dict[str, Any]:
    """
    Load a single result file
    
    Args:
        file_path: Result file path
        
    Returns:
        Dict[str, Any]: Result data dictionary, returns empty dict if loading fails
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Validate required fields
        if 'success_rate' not in data:
            logger.warning(f"Result file missing success_rate field: {file_path}")
            return {}
        
        if 'env_name' not in data:
            logger.warning(f"Result file missing env_name field: {file_path}")
            return {}
        
        return data
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error {file_path}: {e}")
        return {}
    except Exception as e:
        logger.error(f"Failed to load result file {file_path}: {e}")
        return {}


def extract_env_name_from_path(file_path: Path) -> str:
    """
    Extract environment name from file path
    
    Extract environment name from path based on simpler.sh script output directory naming rules
    
    Args:
        file_path: Result file path
        
    Returns:
        str: Extracted environment name
    """
    # Convert path to string and find directory names starting with env_
    path_str = str(file_path)
    
    # Find directory names starting with env_
    parts = path_str.split('/')
    for part in parts:
        if part.startswith('env_'):
            # Remove env_ prefix and octo_init_rng_ suffix
            env_name = part[4:]  # Remove 'env_' prefix
            # Find and remove octo_init_rng_ suffix
            if '_octo_init_rng_' in env_name:
                env_name = env_name.split('_octo_init_rng_')[0]
            return env_name
    
    # If not found, return default value
    return "unknown_env"


def aggregate_results(result_files: List[Path]) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate data from all result files
    
    Args:
        result_files: List of result file paths
        
    Returns:
        Dict[str, Dict[str, Any]]: Aggregated results grouped by environment name
    """
    env_results = defaultdict(list)
    
    for file_path in result_files:
        logger.info(f"Processing file: {file_path}")
        
        # Load result data
        data = load_result_file(file_path)
        if not data:
            continue
        
        # Get environment name
        env_name = data.get('env_name', '')
        if not env_name:
            # If result file doesn't have env_name, try to extract from path
            env_name = extract_env_name_from_path(file_path)
        
        # Collect success rate data
        success_rate = data.get('success_rate', 0.0)
        total_episodes = data.get('total_episodes', 0)
        successful_episodes = data.get('successful_episodes', 0)
        
        # Store result information
        result_info = {
            'success_rate': success_rate,
            'total_episodes': total_episodes,
            'successful_episodes': successful_episodes,
            'file_path': str(file_path),
            'model_type': data.get('model_type', 'unknown'),
            'robot_type': data.get('robot_type', 'unknown'),
            'scene_name': data.get('scene_name', 'unknown')
        }
        
        env_results[env_name].append(result_info)
        logger.debug(f"Environment {env_name}: success rate {success_rate:.3f}")
    
    return dict(env_results)


def calculate_statistics(env_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """
    Calculate statistics for each environment name
    
    Args:
        env_results: Result data grouped by environment name
        
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary containing statistics
    """
    statistics_results = {}
    
    for env_name, results in env_results.items():
        if not results:
            continue
        
        # Extract success rate list
        success_rates = [r['success_rate'] for r in results]
        total_episodes_list = [r['total_episodes'] for r in results]
        successful_episodes_list = [r['successful_episodes'] for r in results]
        
        # Calculate statistics
        num_runs = len(results)
        avg_success_rate = statistics.mean(success_rates)
        median_success_rate = statistics.median(success_rates)
        
        # Calculate standard deviation (if there are multiple results)
        if num_runs > 1:
            std_success_rate = statistics.stdev(success_rates)
        else:
            std_success_rate = 0.0
        
        # Calculate total episode count
        total_episodes = sum(total_episodes_list)
        total_successful_episodes = sum(successful_episodes_list)
        overall_success_rate = total_successful_episodes / total_episodes if total_episodes > 0 else 0.0
        
        # Calculate minimum and maximum values
        min_success_rate = min(success_rates)
        max_success_rate = max(success_rates)
        
        statistics_results[env_name] = {
            'num_runs': num_runs,
            'avg_success_rate': avg_success_rate,
            'median_success_rate': median_success_rate,
            'std_success_rate': std_success_rate,
            'min_success_rate': min_success_rate,
            'max_success_rate': max_success_rate,
            'total_episodes': total_episodes,
            'total_successful_episodes': total_successful_episodes,
            'individual_results': results
        }
        
        logger.info(f"Environment {env_name}: average success rate {avg_success_rate:.3f} ± {std_success_rate:.3f} "
                   f"(runs: {num_runs}, total episodes: {total_episodes})")
    
    return statistics_results


def save_aggregated_results(statistics_results: Dict[str, Dict[str, Any]], 
                          output_file: Path) -> None:
    """
    Save aggregated results to file
    
    Args:
        statistics_results: Statistics results dictionary
        output_file: Output file path
    """
    try:
        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Calculate overall average success rate for all tasks
        all_success_rates = []
        total_episodes = 0
        total_successful_episodes = 0
        
        for stats in statistics_results.values():
            all_success_rates.extend([r['success_rate'] for r in stats['individual_results']])
            total_episodes += stats['total_episodes']
            total_successful_episodes += stats['total_successful_episodes']
        
        # Calculate overall statistics
        overall_avg_success_rate = statistics.mean(all_success_rates) if all_success_rates else 0.0
        overall_std_success_rate = statistics.stdev(all_success_rates) if len(all_success_rates) > 1 else 0.0
        overall_success_rate_by_episodes = total_successful_episodes / total_episodes if total_episodes > 0 else 0.0
        
        # Prepare output data
        output_data = {
            'summary': {
                'total_environments': len(statistics_results),
                'total_runs': sum(stats['num_runs'] for stats in statistics_results.values()),
                'total_episodes': total_episodes,
                'total_successful_episodes': total_successful_episodes,
                'overall_avg_success_rate': overall_avg_success_rate,
                'overall_std_success_rate': overall_std_success_rate,
                'overall_success_rate_by_episodes': overall_success_rate_by_episodes,
                'generation_time': str(Path().cwd()),
                'script_version': '1.0'
            },
            'environment_statistics': statistics_results
        }
        
        # Save to JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Aggregated results saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Failed to save aggregated results: {e}")
        raise


def print_summary_report(statistics_results: Dict[str, Dict[str, Any]]) -> None:
    """
    Print summary report
    
    Args:
        statistics_results: Statistics results dictionary
    """
    print("\n" + "="*80)
    print("SIMPLER ENVIRONMENT EVALUATION RESULTS SUMMARY REPORT")
    print("="*80)
    
    # Sort by average success rate
    sorted_results = sorted(statistics_results.items(), 
                          key=lambda x: x[1]['avg_success_rate'], 
                          reverse=True)
    
    print(f"{'Environment Name':<40} {'Avg Success Rate':<12} {'Std Dev':<10} {'Runs':<8} {'Total Episodes':<12}")
    print("-" * 80)
    
    for env_name, stats in sorted_results:
        print(f"{env_name:<40} {stats['avg_success_rate']:<12.3f} "
              f"{stats['std_success_rate']:<10.3f} {stats['num_runs']:<8} "
              f"{stats['total_episodes']:<12}")
    
    print("-" * 80)
    
    # Calculate overall statistics
    all_success_rates = []
    total_episodes = 0
    total_successful_episodes = 0
    
    for stats in statistics_results.values():
        all_success_rates.extend([r['success_rate'] for r in stats['individual_results']])
        total_episodes += stats['total_episodes']
        total_successful_episodes += stats['total_successful_episodes']
    
    if all_success_rates:
        overall_avg = statistics.mean(all_success_rates)
        overall_std = statistics.stdev(all_success_rates) if len(all_success_rates) > 1 else 0.0
        overall_rate = total_successful_episodes / total_episodes if total_episodes > 0 else 0.0
        
        print(f"{'Overall Statistics':<40} {overall_avg:<12.3f} {overall_std:<10.3f} "
              f"{len(all_success_rates):<8} {total_episodes:<12}")
        print(f"{'Overall Success Rate (by episodes)':<40} {overall_rate:<12.3f}")
        print(f"{'Total Successful Episodes':<40} {total_successful_episodes:<12}")
        print(f"{'All Tasks Average Success Rate':<40} {overall_avg:<12.3f}")
    
    print("="*80)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments
    
    Returns:
        argparse.Namespace: Parsed arguments object
    """
    parser = argparse.ArgumentParser(
        description="Aggregate Simpler environment evaluation results, calculate average success rates for environments with the same name",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--base-dir",
        type=str,
        default="results/",
        help="Base search directory, will recursively search for all result.json files within it"
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        default="aggregated_simpler_results.json",
        help="Output file path for aggregated results"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose log output"
    )
    
    return parser.parse_args()


def main() -> int:
    """
    Main function
    
    Returns:
        int: Program exit code, 0 for success, 1 for failure
    """
    args = parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Convert paths
        base_dir = Path(args.base_dir)
        output_file = Path(args.output_file)
        
        logger.info(f"Starting Simpler evaluation results aggregation...")
        logger.info(f"Search directory: {base_dir}")
        logger.info(f"Output file: {output_file}")
        
        # Find all result files
        result_files = find_result_files(base_dir)
        if not result_files:
            logger.error("No result.json files found")
            return 1
        
        # Aggregate results
        env_results = aggregate_results(result_files)
        if not env_results:
            logger.error("Failed to extract valid data from any result files")
            return 1
        
        # Calculate statistics
        statistics_results = calculate_statistics(env_results)
        
        # Save results
        save_aggregated_results(statistics_results, output_file)
        
        # Print summary report
        print_summary_report(statistics_results)
        
        logger.info("Aggregation completed!")
        return 0
        
    except Exception as e:
        logger.error(f"Error occurred during aggregation: {e}")
        import traceback
        logger.error(f"Detailed error information: {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    exit(main())
