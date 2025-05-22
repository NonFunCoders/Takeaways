"""
Evaluation script for the Takeaways model.

This script evaluates the fine-tuned model on coding benchmarks including 
HumanEval and MBPP.
"""

import os
import sys
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import evaluate
import logging

# Add root directory to path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.settings import MODEL_CONFIG, EVALUATION_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("evaluation.log"),
    ],
)
logger = logging.getLogger(__name__)


def evaluate_humaneval(model, tokenizer, num_samples=EVALUATION_CONFIG["test_sample_size"]):
    """
    Evaluate the model on the HumanEval benchmark.
    
    Args:
        model: Fine-tuned HuggingFace model
        tokenizer: Model tokenizer
        num_samples: Number of samples to evaluate
        
    Returns:
        Dictionary containing evaluation results
    """
    logger.info("Loading HumanEval dataset")
    dataset = load_dataset("openai_humaneval")
    
    if num_samples and num_samples < len(dataset['test']):
        # Take a subset for faster evaluation
        dataset = dataset['test'].select(range(num_samples))
    else:
        dataset = dataset['test']
    
    logger.info(f"Evaluating on {len(dataset)} HumanEval samples")
    
    results = {
        "pass@1": 0,
        "correct_solutions": [],
        "incorrect_solutions": []
    }
    
    for i, sample in enumerate(dataset):
        logger.info(f"Processing sample {i+1}/{len(dataset)}")
        
        # Prepare prompt for the model
        prompt = f"""
Task: Implement the following function
Context: Python Function

{sample['prompt']}
"""
        
        # Generate code with the model
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        
        generated_ids = model.generate(
            input_ids,
            max_length=MODEL_CONFIG["max_length"],
            temperature=EVALUATION_CONFIG["temperature"],
            top_p=0.95,
            do_sample=True,
            num_return_sequences=1,
        )
        
        # Decode the generated response
        response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Extract the generated code (everything after the prompt)
        generated_code = response[len(prompt):].strip()
        
        # Try to extract the code block if present
        if "```python" in generated_code:
            start = generated_code.find("```python") + len("```python")
            end = generated_code.find("```", start)
            if end > start:
                generated_code = generated_code[start:end].strip()
        elif "```" in generated_code:
            start = generated_code.find("```") + len("```")
            end = generated_code.find("```", start)
            if end > start:
                generated_code = generated_code[start:end].strip()
        
        # Check if the solution passes the test
        # This is a simplified placeholder - in a real implementation, 
        # you would run the code against the HumanEval test cases
        check_result = check_code_execution(generated_code, sample)
        
        if check_result["passed"]:
            results["pass@1"] += 1
            results["correct_solutions"].append({
                "task_id": sample["task_id"],
                "generated_code": generated_code,
            })
        else:
            results["incorrect_solutions"].append({
                "task_id": sample["task_id"],
                "generated_code": generated_code,
                "error": check_result.get("error", "Unknown error"),
            })
    
    # Calculate pass@1 rate
    results["pass@1"] = results["pass@1"] / len(dataset)
    
    logger.info(f"HumanEval pass@1: {results['pass@1']:.4f}")
    return results


def check_code_execution(code, sample):
    """
    Check if the generated code passes the test cases.
    
    This is a placeholder function. In a real implementation,
    you would need to safely execute the code against test cases.
    
    Args:
        code: Generated code solution
        sample: HumanEval sample containing test cases
        
    Returns:
        Dictionary with test results
    """
    # This is a placeholder implementation
    # In practice, you would:
    # 1. Create a secure sandbox environment
    # 2. Insert the generated code
    # 3. Run the test cases from the sample
    # 4. Capture the results
    
    logger.warning("Code execution check is a placeholder - not actually running code")
    
    # For demonstration purposes, let's just check for key syntax elements
    # This is NOT a real test - just checks for some syntax patterns
    try:
        # Check if function definition exists
        if sample["entry_point"] in code and "def " in code:
            # This is a very simplistic check - does not actually run the code
            return {"passed": True}
        else:
            return {"passed": False, "error": "Function definition not found"}
    except Exception as e:
        return {"passed": False, "error": str(e)}


def evaluate_code_quality(model, tokenizer, num_samples=10):
    """
    Evaluate the quality of generated code.
    
    Args:
        model: Fine-tuned HuggingFace model
        tokenizer: Model tokenizer
        num_samples: Number of samples to evaluate
        
    Returns:
        Dictionary containing evaluation results
    """
    # This is a placeholder for code quality assessment
    # In a real implementation, you would use metrics like:
    # - Cyclomatic complexity
    # - Maintainability index
    # - Code style compliance
    # - Documentation quality
    
    logger.info("Code quality evaluation not fully implemented")
    
    return {
        "code_quality_score": 0.0,
        "note": "Code quality evaluation is a placeholder"
    }


def main():
    """Main evaluation function."""
    model_path = os.path.join("..", "models", "takeaways")
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        return
    
    logger.info(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    logger.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Run HumanEval evaluation
    humaneval_results = evaluate_humaneval(model, tokenizer)
    
    # Run code quality evaluation
    code_quality_results = evaluate_code_quality(model, tokenizer)
    
    # Combine results
    results = {
        "humaneval": humaneval_results,
        "code_quality": code_quality_results,
    }
    
    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Evaluation results saved to results/evaluation_results.json")
    logger.info(f"HumanEval pass@1: {humaneval_results['pass@1']:.4f}")


if __name__ == "__main__":
    main()
