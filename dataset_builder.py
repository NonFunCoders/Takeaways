import os
import json
from transformers import AutoTokenizer

# Load the tokenizer for Mistral 7B
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

def preprocess_dataset(data_dir, output_dir):
    # Load and preprocess the dataset
    for filename in os.listdir(data_dir):
        if filename.endswith(".jsonl"):
            with open(os.path.join(data_dir, filename), 'r') as f:
                for line in f:
                    sample = json.loads(line)
                    # Tokenize and filter the sample
                    tokenized_sample = tokenizer(sample['instruction'] + sample['input'], return_tensors='pt')
                    # Save the preprocessed sample
                    with open(os.path.join(output_dir, 'preprocessed_dataset.jsonl'), 'a') as output_file:
                        json.dump(tokenized_sample, output_file)
                        output_file.write('\n')

if __name__ == "__main__":
    data_dir = "path_to_your_dataset"
    output_dir = "path_to_output_directory"
    preprocess_dataset(data_dir, output_dir)
