import os
from typing import Optional
import torch
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    AutoTokenizer
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model
)
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TakeawaysTrainer:
    def __init__(
        self,
        base_model: str = "mistralai/Mistral-7B-v0.1",
        output_dir: str = "takeaways-model-output",
    ):
        """Initialize the trainer with model configuration.
        
        Args:
            base_model: Base model to fine-tune
            output_dir: Directory to save model outputs
        """
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure 4-bit quantization
        self.quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        # Configure LoRA
        self.lora_config = LoraConfig(
            r=16,                     # Rank
            lora_alpha=32,            # Alpha scaling
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
    def setup_model(self):
        """Set up the model for training with QLoRA."""
        logger.info(f"Loading base model: {self.base_model}")
        
        # Load base model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=self.quant_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        # Apply LoRA
        model = get_peft_model(model, self.lora_config)
        
        logger.info("Model setup completed")
        return model
        
    def create_training_args(
        self,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
        max_steps: Optional[int] = None
    ) -> TrainingArguments:
        """Create training arguments.
        
        Args:
            num_train_epochs: Number of training epochs
            per_device_train_batch_size: Batch size per device
            gradient_accumulation_steps: Number of steps for gradient accumulation
            learning_rate: Learning rate
            max_steps: Maximum number of training steps (optional)
            
        Returns:
            TrainingArguments object
        """
        return TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            max_steps=max_steps,
            fp16=True,
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=3,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to="tensorboard",
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False
        )
        
    def train(
        self,
        dataset,
        num_train_epochs: int = 3,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
        max_steps: Optional[int] = None
    ):
        """Train the model using QLoRA fine-tuning.
        
        Args:
            dataset: Preprocessed dataset for training
            num_train_epochs: Number of training epochs
            batch_size: Training batch size
            gradient_accumulation_steps: Number of steps for gradient accumulation
            learning_rate: Learning rate
            max_steps: Maximum number of training steps (optional)
        """
        # Setup model
        model = self.setup_model()
        
        # Create training arguments
        training_args = self.create_training_args(
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            max_steps=max_steps
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=None  # Using default data collator
        )
        
        logger.info("Starting training...")
        trainer.train()
        
        # Save the final model
        logger.info("Saving final model...")
        trainer.save_model(str(self.output_dir / "final"))
        
        logger.info("Training completed")
        
    def export_model(self, output_path: Optional[str] = None):
        """Export the trained model for deployment.
        
        Args:
            output_path: Path to save the exported model
        """
        if output_path is None:
            output_path = str(self.output_dir / "exported")
            
        # Load the final trained model
        model = AutoModelForCausalLM.from_pretrained(
            str(self.output_dir / "final"),
            device_map="auto",
            trust_remote_code=True
        )
        
        # Save the model and tokenizer
        model.save_pretrained(output_path)
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        tokenizer.save_pretrained(output_path)
        
        logger.info(f"Model exported to: {output_path}")
