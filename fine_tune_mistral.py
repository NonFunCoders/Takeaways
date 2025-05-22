from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import torch

# Load Mistral 7B model and tokenizer
model_name = "mistralai/Mistral-7B-v0.1"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_4bit=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare model for kbit training
model = prepare_model_for_kbit_training(model)

# Define LoRA configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Create PEFT model
peft_model = get_peft_model(model, lora_config)

# Define training arguments
training_args = {
    "output_dir": "./results",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "logging_steps": 10,
    "save_total_limit": 2,
    "save_steps": 500,
}

# Train the model
# trainer = Trainer(model=peft_model, args=TrainingArguments(**training_args), train_dataset=train_dataset)
# trainer.train()

print("Model fine-tuning setup complete.")
