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
        
    def generate_response(self, prompt: str) -> str:
        """Generate response for a prompt.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated response
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

    def evaluate_bangla_qa(self, num_samples: int = 10) -> Dict[str, float]:
        """Evaluate model on Bangla question answering.
        
        Args:
            num_samples: Number of samples to evaluate
            
        Returns:
            Dictionary with metrics
        """
        logger.info("Starting Bangla QA evaluation...")
        dataset = load_dataset("csebuetnlp/bengali_qa")["test"]
        
        if num_samples:
            dataset = dataset.select(range(num_samples))
            
        correct = 0
        total = len(dataset)
        
        for sample in dataset:
            # Prepare prompt
            prompt = f"### Instruction:\nবাংলায় প্রশ্নের উত্তর দিন।\n\n### Input:\nপ্রসঙ্গ: {sample['context']}\n\nপ্রশ্ন: {sample['question']}\n\n### Response:\n"
            
            # Generate answer
            generated = self.generate_response(prompt)
            
            try:
                answer = generated.split("### Response:\n")[1].strip()
                # Basic exact match scoring
                if answer.lower() == sample['answers']['text'][0].lower():
                    correct += 1
            except (IndexError, KeyError):
                continue
                
        metrics = {
            "bangla_qa_accuracy": correct / total,
            "qa_samples_evaluated": total
        }
        
        logger.info(f"Bangla QA Results: {metrics}")
        return metrics

    def evaluate_bangla_nli(self, num_samples: int = 10) -> Dict[str, float]:
        """Evaluate model on Bangla natural language inference.
        
        Args:
            num_samples: Number of samples to evaluate
            
        Returns:
            Dictionary with metrics
        """
        logger.info("Starting Bangla NLI evaluation...")
        dataset = load_dataset("xnli", "bn")["test"]
        
        if num_samples:
            dataset = dataset.select(range(num_samples))
            
        correct = 0
        total = len(dataset)
        
        label_map = {
            "entailment": "অনুসিদ্ধান্ত",
            "contradiction": "বিরোধিতা",
            "neutral": "নিরপেক্ষ"
        }
        
        for sample in dataset:
            # Prepare prompt
            prompt = f"### Instruction:\nনীচের দুটি বাক্যের মধ্যে সম্পর্ক নির্ণয় করুন: অনুসিদ্ধান্ত, বিরোধিতা, বা নিরপেক্ষ?\n\n### Input:\nবাক্য ১: {sample['premise']}\nবাক্য ২: {sample['hypothesis']}\n\n### Response:\n"
            
            # Generate prediction
            generated = self.generate_response(prompt)
            
            try:
                prediction = generated.split("### Response:\n")[1].strip()
                if prediction == label_map[sample['label']]:
                    correct += 1
            except (IndexError, KeyError):
                continue
                
        metrics = {
            "bangla_nli_accuracy": correct / total,
            "nli_samples_evaluated": total
        }
        
        logger.info(f"Bangla NLI Results: {metrics}")
        return metrics

    def evaluate_bangla_commonsense(self, num_samples: int = 10) -> Dict[str, float]:
        """Evaluate model on Bangla common sense reasoning.
        
        Args:
            num_samples: Number of samples to evaluate
            
        Returns:
            Dictionary with metrics
        """
        logger.info("Starting Bangla Common Sense evaluation...")
        dataset = load_dataset("csebuetnlp/bengali_commonsense")["test"]
        
        if num_samples:
            dataset = dataset.select(range(num_samples))
            
        correct = 0
        total = len(dataset)
        
        for sample in dataset:
            # Prepare prompt
            prompt = f"### Instruction:\nনিম্নলিখিত প্রশ্নের সাধারণ জ্ঞান ভিত্তিক উত্তর দিন।\n\n### Input:\n{sample['question']}\n\n### Response:\n"
            
            # Generate answer
            generated = self.generate_response(prompt)
            
            try:
                prediction = generated.split("### Response:\n")[1].strip()
                if prediction.lower() == sample['answer'].lower():
                    correct += 1
            except (IndexError, KeyError):
                continue
                
        metrics = {
            "bangla_commonsense_accuracy": correct / total,
            "commonsense_samples_evaluated": total
        }
        
        logger.info(f"Bangla Common Sense Results: {metrics}")
        return metrics
        
    def run_full_evaluation(self, 
                          num_samples: int = 10) -> Dict[str, float]:
        """Run complete evaluation suite.
        
        Args:
            num_samples: Number of samples for each evaluation task
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        # Run Bangla QA evaluation
        metrics.update(self.evaluate_bangla_qa(num_samples))
        
        # Run Bangla NLI evaluation
        metrics.update(self.evaluate_bangla_nli(num_samples))
        
        # Run Bangla Common Sense evaluation
        metrics.update(self.evaluate_bangla_commonsense(num_samples))
            
        return metrics
