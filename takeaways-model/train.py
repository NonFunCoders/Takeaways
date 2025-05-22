import argparse
import json
import logging
from pathlib import Path
from typing import Optional
import torch

from data.dataset_loader import DatasetLoader
from model.trainer import TakeawaysTrainer
from evaluation.evaluator import ModelEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path) as f:
        return json.load(f)

def main(args):
    # Load configuration
    config = load_config(args.config)
    
    # Initialize dataset loader
    logger.info("Initializing dataset loader...")
    dataset_loader = DatasetLoader(
        tokenizer_name=config["model"]["tokenizer"]
    )
    
    # Prepare dataset
    logger.info("Preparing training dataset...")
    dataset = dataset_loader.prepare_training_dataset(
        include_code=config["data"].get("include_code_data", False)
    )
    
    # Initialize trainer
    logger.info("Initializing model trainer...")
    trainer = TakeawaysTrainer(
        base_model=config["model"]["base_model"],
        output_dir=args.output_dir
    )
    
    # Train model
    logger.info("Starting model training...")
    trainer.train(
        dataset=dataset,
        num_train_epochs=config["training"]["num_train_epochs"],
        batch_size=config["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=config["training"]["learning_rate"]
    )
    
    # Export model
    logger.info("Exporting trained model...")
    export_path = Path(args.output_dir) / "exported"
    trainer.export_model(str(export_path))
    
    # Run evaluation
    if not args.skip_eval:
        logger.info("Running model evaluation...")
        evaluator = ModelEvaluator(
            model_path=str(export_path),
            temperature=config["evaluation"]["temperature"]
        )
        
        metrics = evaluator.run_full_evaluation(
            num_samples=config["evaluation"]["num_samples"]
        )
        
        # Save metrics
        metrics_path = Path(args.output_dir) / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Evaluation metrics saved to {metrics_path}")
    
    logger.info("Training pipeline completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Takeaways model")
    parser.add_argument(
        "--config",
        type=str,
        default="config/model_config.json",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="takeaways-model-output",
        help="Directory to save model outputs"
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation after training"
    )
    
    args = parser.parse_args()
    main(args)
