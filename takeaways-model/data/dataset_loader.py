from pathlib import Path
from typing import Dict, List, Optional
import json
import logging
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetLoader:
    def __init__(self, tokenizer_name: str = "mistralai/Mistral-7B-v0.1"):
        """Initialize the dataset loader with the specified tokenizer.
        
        Args:
            tokenizer_name: Name or path of the tokenizer to use
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def load_code_alpaca(self, path: Optional[str] = None) -> Dataset:
        """Load and preprocess the Code Alpaca dataset.
        
        Args:
            path: Optional path to local Code Alpaca dataset
            
        Returns:
            Preprocessed dataset
        """
        if path:
            dataset = load_dataset('json', data_files=path)
        else:
            # Load from Hugging Face hub
            dataset = load_dataset('sahil2801/CodeAlpaca-20k')
        
        logger.info(f"Loaded Code Alpaca dataset with {len(dataset['train'])} samples")
        return dataset
        
    def load_human_eval(self) -> Dataset:
        """Load and preprocess the HumanEval dataset.
        
        Returns:
            Preprocessed dataset
        """
        dataset = load_dataset('openai_humaneval')
        logger.info(f"Loaded HumanEval dataset with {len(dataset['test'])} samples")
        return dataset
        
    def load_code_instruct(self, path: Optional[str] = None) -> Dataset:
        """Load and preprocess the CodeInstruct dataset.
        
        Args:
            path: Optional path to local CodeInstruct dataset
            
        Returns:
            Preprocessed dataset
        """
        if path:
            dataset = load_dataset('json', data_files=path)
        else:
            # Load from Hugging Face hub
            dataset = load_dataset('sahil2801/code-instruct-gpt4')
        
        logger.info(f"Loaded CodeInstruct dataset with {len(dataset['train'])} samples")
        return dataset
        
    def preprocess_sample(self, sample: Dict) -> Dict:
        """Preprocess a single dataset sample.
        
        Args:
            sample: Dictionary containing sample data
            
        Returns:
            Preprocessed sample with tokenized inputs
        """
        # Format the text following the Alpaca format
        text = f"### Instruction:\n{sample['instruction']}\n\n"
        if sample.get('input'):
            text += f"### Input:\n{sample['input']}\n\n"
        text += f"### Response:\n{sample['output']}"
        
        # Tokenize
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=2048,
            padding='max_length'
        )
        
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': tokenized['input_ids'].copy()
        }
        
    def merge_datasets(self, datasets: List[Dataset]) -> Dataset:
        """Merge multiple datasets into one.
        
        Args:
            datasets: List of datasets to merge
            
        Returns:
            Merged dataset
        """
        # Combine all datasets
        combined = Dataset.concatenate_datasets(datasets)
        
        # Shuffle the combined dataset
        combined = combined.shuffle(seed=42)
        
        logger.info(f"Created merged dataset with {len(combined)} samples")
        return combined

    def prepare_training_dataset(self, 
                               code_alpaca_path: Optional[str] = None,
                               code_instruct_path: Optional[str] = None) -> Dataset:
        """Prepare the complete training dataset.
        
        Args:
            code_alpaca_path: Optional path to local Code Alpaca dataset
            code_instruct_path: Optional path to local CodeInstruct dataset
            
        Returns:
            Complete preprocessed dataset ready for training
        """
        # Load individual datasets
        datasets = [
            self.load_code_alpaca(code_alpaca_path),
            self.load_human_eval(),
            self.load_code_instruct(code_instruct_path)
        ]
        
        # Merge datasets
        combined = self.merge_datasets(datasets)
        
        # Preprocess all samples
        processed = combined.map(
            self.preprocess_sample,
            remove_columns=combined.column_names,
            num_proc=4
        )
        
        logger.info("Dataset preparation completed")
        return processed
