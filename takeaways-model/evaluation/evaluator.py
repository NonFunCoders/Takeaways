import os
from typing import Dict, List, Optional
import json
import logging
from pathlib import Path
import subprocess
import tempfile
from datasets import load_dataset
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        temperature: float = 0.2,
        max_new_tokens: int = 512
    ):
        """Initialize the evaluator with model and parameters.
        
        Args:
            model_path: Path to the trained model
            tokenizer_path: Path to tokenizer (defaults to model_path)
            temperature: Sampling temperature
            max_new_tokens: Maximum new tokens to generate
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        
        # Load model and tokenizer
        logger.info("Loading model and tokenizer...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        
    def generate_code(self, prompt: str) -> str:
        """Generate code from a prompt.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated code
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
    def evaluate_human_eval(self, num_samples: int = 10) -> Dict[str, float]:
        """Evaluate model on HumanEval benchmark.
        
        Args:
            num_samples: Number of samples to evaluate
            
        Returns:
            Dictionary with metrics
        """
        logger.info("Starting HumanEval evaluation...")
        dataset = load_dataset("openai_humaneval")["test"]
        
        if num_samples:
            dataset = dataset.select(range(num_samples))
            
        correct = 0
        total = len(dataset)
        
        for sample in dataset:
            # Prepare prompt
            prompt = f"### Instruction:\nImplement the following Python function.\n\n### Input:\n{sample['prompt']}\n\n### Response:\n"
            
            # Generate solution
            generated_code = self.generate_code(prompt)
            
            # Extract just the function implementation
            try:
                generated_code = generated_code.split("### Response:\n")[1].strip()
            except IndexError:
                continue
                
            # Write test to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(generated_code + "\n\n" + sample['test'])
                test_file = f.name
                
            # Run tests
            try:
                subprocess.run(['python', test_file], check=True, capture_output=True)
                correct += 1
            except subprocess.CalledProcessError:
                pass
            finally:
                os.unlink(test_file)
                
        metrics = {
            "human_eval_accuracy": correct / total,
            "samples_evaluated": total
        }
        
        logger.info(f"HumanEval Results: {metrics}")
        return metrics
        
    def evaluate_mbpp(self, num_samples: int = 10) -> Dict[str, float]:
        """Evaluate model on MBPP benchmark.
        
        Args:
            num_samples: Number of samples to evaluate
            
        Returns:
            Dictionary with metrics
        """
        logger.info("Starting MBPP evaluation...")
        dataset = load_dataset("mbpp")["test"]
        
        if num_samples:
            dataset = dataset.select(range(num_samples))
            
        correct = 0
        total = len(dataset)
        
        for sample in dataset:
            # Prepare prompt
            prompt = (
                f"### Instruction:\n{sample['text']}\n\n"
                f"### Input:\nWrite a Python function to solve this problem.\n\n"
                f"### Response:\n"
            )
            
            # Generate solution
            generated_code = self.generate_code(prompt)
            
            # Extract just the function implementation
            try:
                generated_code = generated_code.split("### Response:\n")[1].strip()
            except IndexError:
                continue
                
            # Write test file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(generated_code + "\n\n")
                # Add test cases
                for test_case in sample['test_list']:
                    f.write(f"assert {test_case}\n")
                test_file = f.name
                
            # Run tests
            try:
                subprocess.run(['python', test_file], check=True, capture_output=True)
                correct += 1
            except subprocess.CalledProcessError:
                pass
            finally:
                os.unlink(test_file)
                
        metrics = {
            "mbpp_accuracy": correct / total,
            "samples_evaluated": total
        }
        
        logger.info(f"MBPP Results: {metrics}")
        return metrics
        
    def evaluate_code_explanation(self, samples: List[Dict[str, str]]) -> Dict[str, float]:
        """Evaluate model's code explanation capabilities.
        
        Args:
            samples: List of dictionaries with 'code' and 'explanation' keys
            
        Returns:
            Dictionary with metrics
        """
        logger.info("Evaluating code explanation capabilities...")
        
        scores = []
        for sample in samples:
            # Prepare prompt
            prompt = f"### Instruction:\nExplain the following code step by step.\n\n### Input:\n{sample['code']}\n\n### Response:\n"
            
            # Generate explanation
            generated_explanation = self.generate_code(prompt)
            
            # Compare with reference explanation (simple word overlap metric)
            reference_words = set(sample['explanation'].lower().split())
            generated_words = set(generated_explanation.lower().split())
            
            overlap = len(reference_words.intersection(generated_words)) / len(reference_words)
            scores.append(overlap)
            
        metrics = {
            "explanation_quality": np.mean(scores),
            "samples_evaluated": len(samples)
        }
        
        logger.info(f"Code Explanation Results: {metrics}")
        return metrics
        
    def run_full_evaluation(self, 
                          num_samples: int = 10,
                          explanation_samples: Optional[List[Dict[str, str]]] = None) -> Dict[str, float]:
        """Run complete evaluation suite.
        
        Args:
            num_samples: Number of samples for HumanEval and MBPP
            explanation_samples: Optional samples for explanation evaluation
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        # Run HumanEval
        metrics.update(self.evaluate_human_eval(num_samples))
        
        # Run MBPP
        metrics.update(self.evaluate_mbpp(num_samples))
        
        # Run explanation evaluation if samples provided
        if explanation_samples:
            metrics.update(self.evaluate_code_explanation(explanation_samples))
            
        return metrics
