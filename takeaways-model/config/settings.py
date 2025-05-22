"""
Configuration settings for the Takeaways model.

This file contains the default configuration settings for model training,
dataset processing, and evaluation.
"""

# Base model settings
MODEL_CONFIG = {
    "base_model": "mistralai/Mistral-7B-v0.1",
    "model_type": "causal_lm",
    "max_length": 2048,
    "lora_r": 8,             # LoRA rank
    "lora_alpha": 32,        # LoRA alpha
    "lora_dropout": 0.1,     # LoRA dropout
    # Target modules for LoRA
    "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}

# Training settings
TRAINING_CONFIG = {
    "output_dir": "./models/takeaways",
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "num_train_epochs": 3,
    "warmup_steps": 100,
    "fp16": True,
    "logging_steps": 10,
    "save_strategy": "epoch",
    "evaluation_strategy": "steps",
    "eval_steps": 200,
    "save_total_limit": 3,
}

# Dataset settings
DATASET_CONFIG = {
    "datasets": ["code_alpaca", "human_eval", "code_instruct"],
    "data_dir": "./data",
    "train_split": 0.9,
    "eval_split": 0.1,
    "processed_data_path": "./data/processed",
}

# Evaluation settings
EVALUATION_CONFIG = {
    "metrics": ["humaneval_pass@1", "mbpp_score", "code_quality"],
    "test_sample_size": 100,
    "top_k": 5,
    "temperature": 0.7,
}

# Deployment settings
DEPLOYMENT_CONFIG = {
    "local": {
        "model_path": "./models/takeaways",
        "quantize": True,
        "port": 5000,
    },
    "web": {
        "api_host": "0.0.0.0",
        "api_port": 8000,
        "max_request_length": 4096,
        "max_response_length": 2048,
        "streaming": True,
    },
}
