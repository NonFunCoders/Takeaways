"""
Training script for the Takeaways model.

This script handles the fine-tuning of the base model (Mistral 7B) using QLoRA
for parameter-efficient training.
"""

import os
import sys
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)
from datasets import load_dataset, Dataset, concatenate_datasets
import bitsandbytes as bnb
from tqdm import tqdm
import sys
import logging

# Add root directory to path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.settings import MODEL_CONFIG, TRAINING_CONFIG, DATASET_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(TRAINING_CONFIG["output_dir"], "training.log")),
    ],
)
logger = logging.getLogger(__name__)


def load_tokenized_dataset(tokenizer, dataset_path=None):
    """
    Load and tokenize dataset.
    
    Args:
        tokenizer: HuggingFace tokenizer
        dataset_path: Path to preprocessed dataset (optional)
    
    Returns:
        Tokenized dataset split into train and validation sets
    """
    if dataset_path and os.path.exists(dataset_path):
        logger.info(f"Loading preprocessed dataset from {dataset_path}")
        dataset = load_dataset("json", data_files=dataset_path)
    else:
        # This is a simplified example - in practice, you would use
        # the DatasetLoader class to load and preprocess datasets
        logger.info("Loading datasets from Hugging Face")
        datasets = []
        
        # Load CodeAlpaca
        try:
            code_alpaca = load_dataset("sahil2801/CodeAlpaca-20k")
            datasets.append(code_alpaca)
            logger.info("Loaded CodeAlpaca dataset")
        except Exception as e:
            logger.error(f"Error loading CodeAlpaca: {e}")
        
        # Combine datasets
        if datasets:
            dataset = concatenate_datasets(datasets)
        else:
            raise ValueError("No datasets were successfully loaded")
    
    # Tokenize dataset
    def preprocess_function(examples):
        # Format as instruction, input, output
        prompt = f"""Task: {examples['instruction']}
Context: {examples['input'] if 'input' in examples and examples['input'] else 'None'}
Code:
```
{examples['output']}
```"""
        
        return tokenizer(
            prompt, 
            truncation=True,
            max_length=MODEL_CONFIG["max_length"],
            padding="max_length",
        )
    
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=4,
        remove_columns=dataset.column_names,
    )
    
    # Split into train and validation sets
    tokenized_dataset = tokenized_dataset["train"].train_test_split(
        test_size=DATASET_CONFIG["eval_split"]
    )
    
    return tokenized_dataset


def train():
    """Main training function."""
    # Create output directory if it doesn't exist
    os.makedirs(TRAINING_CONFIG["output_dir"], exist_ok=True)
    
    logger.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CONFIG["base_model"],
        use_fast=True,
    )
    
    # Ensure tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info("Loading model")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_CONFIG["base_model"],
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_4bit=True,
        quantization_config=bnb.nn.modules.Linear4bit.quantize()
    )
    
    # Prepare model for training
    logger.info("Preparing model for training")
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=MODEL_CONFIG["lora_r"],
        lora_alpha=MODEL_CONFIG["lora_alpha"],
        lora_dropout=MODEL_CONFIG["lora_dropout"],
        target_modules=MODEL_CONFIG["lora_target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    logger.info(f"Trainable parameters: {model.print_trainable_parameters()}")
    
    # Load dataset
    logger.info("Loading and tokenizing dataset")
    dataset_path = os.path.join(DATASET_CONFIG["processed_data_path"], "processed_datasets.json")
    tokenized_dataset = load_tokenized_dataset(tokenizer, dataset_path)
    
    # Configure training arguments
    training_args = TrainingArguments(
        **TRAINING_CONFIG,
        report_to="none",  # Disable wandb for now
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        ),
    )
    
    # Train model
    logger.info("Starting training")
    trainer.train()
    
    # Save model
    logger.info(f"Saving model to {TRAINING_CONFIG['output_dir']}")
    trainer.save_model()
    tokenizer.save_pretrained(TRAINING_CONFIG["output_dir"])
    
    logger.info("Training complete")


if __name__ == "__main__":
    train()
