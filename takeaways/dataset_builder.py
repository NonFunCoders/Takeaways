import json
from typing import List, Dict

class DatasetBuilder:
    def __init__(self, config):
        self.config = config
        
    def collect_data(self) -> List[Dict[str, str]]:
        """Collect raw data from various sources"""
        # Placeholder for data collection logic
        return [
            {
                "input": "Write a function to calculate the sum of two numbers.",
                "output": "def calculate_sum(a, b):\n    return a + b"
            }
        ]
        
    def preprocess_data(self, raw_data: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Preprocess raw data into a format suitable for training"""
        processed_data = []
        for entry in raw_data:
            processed_entry = {
                "input": f"```python\n{entry['input']}\n```",
                "output": f"```python\n{entry['output']}\n```"
            }
            processed_data.append(processed_entry)
        return processed_data
        
    def save_dataset(self, dataset: List[Dict[str, str]], path: str):
        """Save the preprocessed dataset to disk"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2)