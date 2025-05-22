"""
Dataset processing script for Takeaways model.

This script handles the processing of various coding datasets used for training
and evaluation of the Takeaways model.
"""

import os
import sys
import json
import logging
import argparse
from datasets import load_dataset, Dataset, concatenate_datasets

# Add root directory to path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.settings import DATASET_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("dataset_processing.log"),
    ],
)
logger = logging.getLogger(__name__)


def load_code_alpaca():
    """Load and process the Code Alpaca dataset."""
    logger.info("Loading Code Alpaca dataset")
    
    try:
        dataset = load_dataset("sahil2801/CodeAlpaca-20k")
        
        # Map to instruction format
        def format_alpaca(example):
            return {
                "instruction": example["instruction"],
                "input": example.get("input", ""),
                "output": example["output"]
            }
        
        return dataset.map(format_alpaca)
    except Exception as e:
        logger.error(f"Error loading Code Alpaca: {e}")
        return None


def load_human_eval():
    """Load and process the HumanEval dataset."""
    logger.info("Loading HumanEval dataset")
    
    try:
        dataset = load_dataset("openai_humaneval")
        
        # Convert to instruction format
        def format_humaneval(example):
            # Extract the function signature and docstring as instruction
            prompt_parts = example["prompt"].split("\n\n")
            docstring = ""
            
            if len(prompt_parts) > 1:
                docstring = prompt_parts[1].strip()
            
            return {
                "instruction": f"Implement the following function: {example['entry_point']}\n\n{docstring}",
                "input": example["prompt"],
                "output": example["canonical_solution"],
                "test": example["test"]  # Keep test cases for evaluation
            }
        
        return dataset.map(format_humaneval)
    except Exception as e:
        logger.error(f"Error loading HumanEval: {e}")
        return None


def load_code_instruct():
    """Load and process the CodeInstruct dataset."""
    logger.info("Loading CodeInstruct dataset")
    
    # This is a placeholder as CodeInstruct might not be directly available
    # In a real implementation, you would need to adapt this to the actual dataset
    try:
        # Simulated dataset - replace this with actual implementation
        logger.warning("CodeInstruct dataset loader is a placeholder")
        
        return None
    except Exception as e:
        logger.error(f"Error loading CodeInstruct: {e}")
        return None


def combine_datasets(datasets):
    """Combine multiple datasets into one."""
    valid_datasets = [d for d in datasets if d is not None]
    
    if not valid_datasets:
        logger.error("No valid datasets to combine")
        return None
    
    if len(valid_datasets) == 1:
        return valid_datasets[0]
    
    # Harmonize features across datasets to ensure consistent column names
    # This is a simplified approach - in practice you may need more complex feature alignment
    common_columns = ["instruction", "input", "output"]
    harmonized_datasets = []
    
    for ds in valid_datasets:
        # Ensure dataset has all required columns
        if all(col in ds.column_names for col in common_columns):
            # Extract only the common columns
            harmonized_ds = ds.select_columns(common_columns)
            harmonized_datasets.append(harmonized_ds)
        else:
            logger.warning(f"Dataset skipped due to missing required columns")
    
    if not harmonized_datasets:
        logger.error("No datasets remained after harmonizing columns")
        return None
    
    # Concatenate all datasets
    logger.info(f"Combining {len(harmonized_datasets)} datasets")
    combined_dataset = concatenate_datasets(harmonized_datasets)
    
    return combined_dataset


def save_processed_dataset(dataset, output_path):
    """Save the processed dataset to disk."""
    if dataset is None:
        logger.error("No dataset to save")
        return False
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save dataset in jsonl format
        dataset.to_json(output_path)
        
        logger.info(f"Dataset saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving dataset: {e}")
        return False


def main():
    """Main function to process datasets."""
    parser = argparse.ArgumentParser(description="Process coding datasets for Takeaways model")
    parser.add_argument("--output", type=str, default=os.path.join(DATASET_CONFIG["processed_data_path"], "processed_datasets.json"),
                        help="Output path for the processed datasets")
    args = parser.parse_args()
    
    # Load datasets
    datasets = []
    
    if "code_alpaca" in DATASET_CONFIG["datasets"]:
        code_alpaca_dataset = load_code_alpaca()
        if code_alpaca_dataset:
            datasets.append(code_alpaca_dataset)
    
    if "human_eval" in DATASET_CONFIG["datasets"]:
        human_eval_dataset = load_human_eval()
        if human_eval_dataset:
            datasets.append(human_eval_dataset)
    
    if "code_instruct" in DATASET_CONFIG["datasets"]:
        code_instruct_dataset = load_code_instruct()
        if code_instruct_dataset:
            datasets.append(code_instruct_dataset)
    
    # Combine datasets
    combined_dataset = combine_datasets(datasets)
    
    if combined_dataset:
        logger.info(f"Combined dataset size: {len(combined_dataset)} examples")
        
        # Save processed dataset
        save_processed_dataset(combined_dataset, args.output)
    else:
        logger.error("Failed to create combined dataset")


if __name__ == "__main__":
    main()
