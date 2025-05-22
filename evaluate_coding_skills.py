import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the fine-tuned model and tokenizer
model_name = "path_to_your_fine_tuned_model"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def evaluate_human_eval():
    # Load HumanEval dataset
    human_eval_dataset = evaluate.load_dataset("humaneval")
    
    # Evaluate the model on HumanEval
    results = []
    for sample in human_eval_dataset:
        # Generate code using the model
        input_ids = tokenizer(sample['prompt'], return_tensors='pt').input_ids
        generated_ids = model.generate(input_ids, max_length=512)
        generated_code = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Check if the generated code passes the test cases
        # This part is simplified and assumes you have a function to check the code against test cases
        result = check_code_against_test_cases(generated_code, sample['test_cases'])
        results.append(result)
    
    # Calculate the pass rate
    pass_rate = sum(results) / len(results)
    print(f"HumanEval pass rate: {pass_rate}")

def evaluate_mbpp():
    # Load MBPP dataset
    mbpp_dataset = evaluate.load_dataset("mbpp")
    
    # Evaluate the model on MBPP
    results = []
    for sample in mbpp_dataset:
        # Generate code using the model
        input_ids = tokenizer(sample['prompt'], return_tensors='pt').input_ids
        generated_ids = model.generate(input_ids, max_length=512)
        generated_code = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Check if the generated code is correct
        # This part is simplified and assumes you have a function to check the code correctness
        result = check_code_correctness(generated_code, sample['test_cases'])
        results.append(result)
    
    # Calculate the accuracy
    accuracy = sum(results) / len(results)
    print(f"MBPP accuracy: {accuracy}")

def self_consistency_check():
    # Generate multiple outputs for the same prompt and check consistency
    prompt = "Write a function to check for balanced parentheses."
    results = []
    for _ in range(5):
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids
        generated_ids = model.generate(input_ids, max_length=512)
        generated_code = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        results.append(generated_code)
    
    # Check consistency among the generated codes
    consistency = len(set(results)) == 1
    print(f"Self-consistency: {consistency}")

if __name__ == "__main__":
    evaluate_human_eval()
    evaluate_mbpp()
    self_consistency_check()
