"""
Base Evaluator Abstract Class

Defines a unified evaluation interface that all specific environment evaluators should inherit from.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import logging
import time
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf
logger = logging.getLogger(__name__)


class BaseEvaluator(ABC):
    
    def __init__(self, config: OmegaConf, output_structure: Dict[str, Path]):
        """
        Initialize the evaluator
        
        Args:
            config: OmegaConf configuration object
            output_structure: Dictionary containing output structure
        """
        self.config = config
        self.output_structure = output_structure
        self.results = {}
        self.start_time = None
        self.end_time = None
   
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = self.config.get("log_level", "INFO")
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    @abstractmethod
    def setup_environment(self) -> Any:
        """
        Setup evaluation environment
        """
        pass

    @abstractmethod
    def setup_model(self) -> Any:
        """
        Setup evaluation model
        
        Returns:
            Any: Model object
        """
        pass
    
    def run_evaluation(self) -> Dict[str, Any]:
        """
        Run complete evaluation process
        
        Returns:
            Dict[str, Any]: Evaluation results summary
        """
        logger.info("Starting evaluation...")
        self.start_time = time.time()

        self.env = self.setup_environment()
        self.model = self.setup_model()

        try:
            # Subclasses implement specific evaluation logic
            self.results = self._run_evaluation_impl()
            
        except Exception as e:
            logger.error(f"Error occurred during evaluation: {e}")
            raise
        finally:
            self.end_time = time.time()
        
        # Record evaluation time (if these fields are not already in results)
        if "evaluation_time" not in self.results:
            self.results["evaluation_time"] = self.end_time - self.start_time
        if "start_time" not in self.results:
            self.results["start_time"] = self.start_time
        if "end_time" not in self.results:
            self.results["end_time"] = self.end_time
        
        logger.info(f"Evaluation completed, time taken: {self.results['evaluation_time']:.2f} seconds")
        return self.results
    
    @abstractmethod
    def _run_evaluation_impl(self) -> Dict[str, Any]:
        """
        Implement specific evaluation logic
        
        Returns:
            Dict[str, Any]: Evaluation results
        """
        pass