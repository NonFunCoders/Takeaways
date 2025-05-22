class ModelConfig:
    base_model = "deepseek/deepseek-r1-distill-qwen-32b:free"
    dataset_path = "data/code_dataset.json"
    output_dir = "models/takeaways"
    learning_rate = 1e-5
    weight_decay = 0.1
    batch_size = 8
    num_epochs = 3
    max_seq_length = 1024
    gradient_checkpointing = True