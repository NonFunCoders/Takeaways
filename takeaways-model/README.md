# Takeaways Model

An advanced AI coding assistant built on Mistral 7B, specialized in providing high-quality code solutions with detailed explanations.

## Features

- Multi-step reasoning with Chain-of-Thought
- Code generation and explanation
- Real-time interaction support
- Structured MDX-style outputs
- Local deployment via Ollama
- Web API interface

## Project Structure

```
takeaways-model/
├── data/               # Dataset processing and management
├── model/             # Model training and fine-tuning
├── evaluation/        # Testing and benchmarking
├── serve/             # Deployment and API
├── scripts/           # Utility scripts
└── config/            # Configuration files
```

## Setup

1. Install dependencies
```bash
pip install -r requirements.txt
```

2. Prepare the model
```bash
ollama create takeaways
```

3. Run local interface
```bash
python serve/local.py
```

4. Start web API
```bash
python serve/api.py
```

## Model Architecture

- Base: Mistral 7B
- Fine-tuning: QLoRA with 4-bit quantization
- Training Data: CodeAlpaca, HumanEval, CodeInstruct
- Evaluation: MBPP and HumanEval benchmarks

## Usage

### Local CLI
```bash
takeaways "Write a function to implement binary search"
```

### API Endpoint
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a function to implement binary search"}'
```

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.
