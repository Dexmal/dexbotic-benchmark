"""
Calvin Environment Evaluator
"""

import os
import json
from pathlib import Path
from typing import Dict, Any
import logging
import time
import numpy as np
import tqdm
import hydra
import torch
import torchvision
from omegaconf import OmegaConf, ListConfig
from pytorch_lightning import seed_everything
from termcolor import colored
import shutil
import imageio
from collections import Counter, defaultdict
from evaluation.utils.tools import resize_frames_for_video

from .base_evaluator import BaseEvaluator

from calvin.calvin_models.calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin.calvin_models.calvin_agent.evaluation.utils import (
    collect_plan,
    count_success,
    create_tsne,
    get_env_state_for_initial_condition,
    join_vis_lang,
)

logger = logging.getLogger(__name__)


EP_LEN = 360
NUM_SEQUENCES = int(os.environ.get("CALVIN_NUM_SEQUENCES", 1000))

def load_dataset_statistics(val_dataset_dir, transforms):
    """
    Args:
        train_dataset_dir: path of the training folder
        val_dataset_dir: path of the validation folder
        transforms: transforms loaded from hydra conf

    Returns:
        transforms: potentially updated transforms
    """
    try:
        statistics = OmegaConf.load(Path(val_dataset_dir) / "statistics.yaml")
        statistics = OmegaConf.create(OmegaConf.to_yaml(statistics).replace("calvin_models.", ""))
        for modality in transforms:
            if modality in statistics:
                conf_transforms = transforms[modality]
                dataset_transforms = statistics[modality]
                for dataset_trans in dataset_transforms:
                    exists = False
                    for i, conf_trans in enumerate(conf_transforms):
                        if dataset_trans["_target_"] == conf_trans["_target_"]:
                            exists = True
                            transforms[modality][i] = dataset_trans
                            break
                    if not exists:
                        transforms[modality] = ListConfig([*conf_transforms, dataset_trans])
    except FileNotFoundError:
        logger.warning("Could not load statistics.yaml")

    return transforms

class CalvinEvaluator(BaseEvaluator):
    """
    Calvin Environment Evaluator
    """
    
    def setup_environment(self) -> Any:
        """Setup Calvin environment"""
        # Set random seed
        seed_everything(self.config.seed, workers=True)
        
        lang_folder = self.config.datamodule.datasets.lang_dataset.lang_folder
        # Must use relative path for initialization
        if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
            hydra.initialize("../../calvin/calvin_models/conf/datamodule/datasets")
        # We don't want to use shm dataset for evaluation
        datasets_cfg = hydra.compose("vision_lang.yaml", overrides=["lang_dataset.lang_folder=" + lang_folder])
        # Since we don't use the trainer during inference, manually set up data_module
        self.config.datamodule.datasets = datasets_cfg
        
        dataset_path = self.config.dataset_path
        if not os.path.isabs(dataset_path):
            current_dir = Path(__file__).parent.parent.parent
            dataset_path = str(current_dir / dataset_path)
        self.config.datamodule.root_data_dir = dataset_path

        class ValDataset_Args:
            def __init__(self, config):
                self.abs_datasets_dir = config.datamodule.root_data_dir
                self.observation_space = config.datamodule.observation_space
                val_transforms_statistics = load_dataset_statistics(self.abs_datasets_dir, config.datamodule.transforms['val'])
                val_transforms_dict = {
                    cam: [hydra.utils.instantiate(transform) for transform in val_transforms_statistics[cam]] for cam in val_transforms_statistics
                }
                self.transforms = {key: torchvision.transforms.Compose(val) for key, val in val_transforms_dict.items()}
                self.proprio_state = config.datamodule.datasets['lang_dataset'].proprio_state

        val_dataset = ValDataset_Args(self.config)
        device = torch.device(f"cuda:{self.config.device}")

        # Use absolute path to load configuration file
        current_dir = Path(__file__).parent.parent.parent
        rollout_cfg_path = current_dir / "calvin" / "calvin_models" / "conf" / "callbacks" / "rollout" / "default.yaml"
        rollout_cfg = OmegaConf.load(str(rollout_cfg_path))
        env = hydra.utils.instantiate(rollout_cfg.env_cfg, val_dataset, device, show_gui=False)

        return env
    
    def setup_model(self) -> Any:
        """Setup VLA agent"""
        try:   
            # Use VLA agent
            from evaluation.policies.calvin_vla_agent import CalvinVLAAgent
            
            logger.info("Loading VLA agent")
            model = CalvinVLAAgent(self.config)
            
            return model
            
        except ImportError as e:
            logger.error(f"Unable to import VLA agent: {e}")
            raise
    
    def _run_evaluation_impl(self) -> Dict[str, Any]:
        """Implement Calvin evaluation logic"""
        try:            
            # Get configuration parameters
            dataset_path = self.config.dataset_path
            debug = self.config.debug
            create_plan_tsne = self.config.get("create_plan_tsne", False)
            
            logger.info(f"Starting Calvin evaluation, dataset path: {dataset_path}")
            
            # Use absolute path to load task configuration
            current_dir = Path(__file__).parent.parent.parent
            task_cfg_path = current_dir / "calvin" / "calvin_models" / "conf" / "callbacks" / "rollout" / "tasks" / "new_playtable_tasks.yaml"
            val_annotations_path = current_dir / "calvin" / "calvin_models" / "conf" / "annotations" / "new_playtable_validation.yaml"
            
            task_cfg = OmegaConf.load(str(task_cfg_path))
            task_oracle = hydra.utils.instantiate(task_cfg)
            val_annotations = OmegaConf.load(str(val_annotations_path))
            
            # Get evaluation sequences
            eval_sequences = get_sequences(NUM_SEQUENCES)
            
            # Initialize result collection
            results = []
            plans = defaultdict(list)
            
            if not debug:
                eval_sequences = tqdm.tqdm(eval_sequences, position=0, leave=True)
            
            # Start evaluation
            for i, (initial_state, eval_sequence) in enumerate(eval_sequences):
                # Create base video path for each sequence
                base_video_path = Path(self.output_structure['videos_dir']) / f"sequence_{i}"
                base_video_path.mkdir(parents=True, exist_ok=True)

                result = self._evaluate_sequence(
                    self.env, self.model, task_oracle, initial_state, eval_sequence, 
                    val_annotations, plans, debug, base_video_path
                )
                results.append(result)
                
                if not debug:
                    eval_sequences.set_description(
                        " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(count_success(results))]) + "|"
                    )
            
            # Create TSNE plot (if needed)
            if create_plan_tsne:
                create_tsne(plans, self.output_structure['logs_dir'], 0)
            
            # Print and save results
            results = self.print_and_save(results, eval_sequences)
            
            return results
            
        except Exception as e:
            logger.error(f"Error occurred during Calvin evaluation: {e}")
            return {
                "error": str(e),
                "total_sequences": 0,
                "successful_sequences": 0,
                "overall_success_rate": 0.0,
                "step_success_rates": [0.0] * 5,
                "detailed_results": []
            }
    
    def print_and_save(self, results, sequences):
        avg_seq_len = np.mean(results)
        chain_sr = {i + 1: sr for i, sr in enumerate(count_success(results))}
        logger.info(f"Average successful sequence length: {avg_seq_len}")
        logger.info("Success rates for i instructions in a row:")
        for i, sr in chain_sr.items():
            logger.info(f"{i}: {sr * 100:.1f}%")
            
        cnt_success = Counter()
        cnt_fail = Counter()

        for result, (_, sequence) in zip(results, sequences):
            for successful_tasks in sequence[:result]:
                cnt_success[successful_tasks] += 1
            if result < len(sequence):
                failed_task = sequence[result]
                cnt_fail[failed_task] += 1

        total = cnt_success + cnt_fail
        task_info = {}
        for task in total:
            task_info[task] = {"success": cnt_success[task], "total": total[task]}
            logger.info(f"{task}: {cnt_success[task]} / {total[task]} |  SR: {cnt_success[task] / total[task] * 100:.1f}%")

        data = {"avg_seq_len": avg_seq_len, "chain_sr": chain_sr, "task_info": task_info}

        for k, v in data.items():
            logger.info(f"{k}: {v}")
        
        return data

    def _evaluate_sequence(self, env, model, task_checker, initial_state, eval_sequence, 
                          val_annotations, plans, debug, video_path=None):
        """
        Evaluate language instruction sequence
        
        Args:
            env: Calvin environment
            model: Model
            task_checker: Task checker
            initial_state: Initial state
            eval_sequence: Evaluation sequence
            val_annotations: Validation annotations
            plans: Plan collector
            debug: Debug mode
            video_path: Video path
            
        Returns:
            int: Number of successfully completed steps
        """
        # Set environment initial state
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

        success_counter = 0
        
        if debug:
            time.sleep(1)
            logger.info()
            logger.info()
            logger.info(f"Evaluation sequence: {' -> '.join(eval_sequence)}")
            logger.info("Subtasks: ", end="")
            
        for i, subtask in enumerate(eval_sequence):
            if video_path:
                # Save video file directly in sequence folder
                seq_video_path = video_path / f"step_{i}_{subtask}.mp4"
            else:
                seq_video_path = None
                
            success = self._rollout(
                env, model, task_checker, subtask, val_annotations, 
                plans, debug, seq_video_path
            )
            
            if success:
                if video_path and seq_video_path:
                    success_video_path = video_path / f"step_{i}_{subtask}_success.mp4"
                    shutil.move(seq_video_path, success_video_path)
                success_counter += 1
            else:
                if video_path and seq_video_path:
                    fail_video_path = video_path / f"step_{i}_{subtask}_fail.mp4"
                    shutil.move(seq_video_path, fail_video_path)
                return success_counter
                
        return success_counter
    
    def _rollout(self, env, model, task_oracle, subtask, val_annotations, 
                plans, debug, video_path=None, macro_block_size=16):
        """
        Run actual rollout on a single subtask
        
        Args:
            env: Calvin environment
            model: Model
            task_oracle: Task oracle
            subtask: Subtask
            val_annotations: Validation annotations
            plans: Plan collector
            debug: Debug mode
            video_path: Video path
            
        Returns:
            bool: Whether the task was completed successfully
        """
        if debug:
            logger.info(f"{subtask} ", end="")
            time.sleep(0.5)
            
        obs = env.get_obs()
        
        # Get language annotation for subtask
        lang_annotation = val_annotations[subtask][0]
        
        # Reset model
        model.reset()
        start_info = env.get_info()
        
        # Initialize video recording
        frames = []  # Store video frames

        # Execute rollout
        episode_first_frame = True
        for step in range(EP_LEN):
            obs['task'] = subtask
            action = model.step(obs, lang_annotation, episode_first_frame=episode_first_frame)
            obs, _, _, current_info = env.step(action)
            episode_first_frame = False
            
            # Record video
            if video_path:
                img = env.render(mode="rgb_array")
                frames.append(img)  # Store RGB image directly
                
            # Debug mode display
            if debug:
                img = env.render(mode="rgb_array")
                join_vis_lang(img, lang_annotation)
                
            # Collect plan data (first step only)
            if step == 0:
                collect_plan(model, plans, subtask)

            # Check if current step solved the task
            current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
            if len(current_task_info) > 0:
                if debug:
                    logger.info(colored("success", "green"), end=" ")
                if video_path and frames:
                    resized_frames = resize_frames_for_video(frames, macro_block_size)
                    imageio.mimsave(video_path, resized_frames, fps=15)
                return True
                
        if debug:
            logger.info(colored("fail", "red"), end=" ")
        if video_path and frames:
            resized_frames = resize_frames_for_video(frames, macro_block_size)
            imageio.mimsave(video_path, resized_frames, fps=15)
        return False
