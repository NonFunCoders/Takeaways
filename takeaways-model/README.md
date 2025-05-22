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

## Automated Training with GitHub Actions

The project includes a GitHub Actions workflow for automated model training and publishing. The workflow:

1. Trains the model using GPU-enabled runners
2. Evaluates model performance
3. Publishes the trained model to Hugging Face Hub
4. Creates a GitHub release with training metrics

### Setup Requirements

1. Configure GitHub Secrets:
   - `HUGGING_FACE_TOKEN`: Access token from Hugging Face (create at https://huggingface.co/settings/tokens)
   - `GITHUB_TOKEN`: Automatically provided by GitHub Actions

2. Configure GPU Runner:
   The workflow requires a self-hosted runner with GPU access. To set up:
   
   ```bash
   # Add self-hosted runner with GPU
   gh runner create --name gpu-runner --labels gpu
   ```
   
   Alternatively, modify `.github/workflows/train-and-publish.yml` to use cloud GPU services.

3. Create Hugging Face Repository:
   - Create a new model repository at https://huggingface.co/new
   - Name it 'Takeaways' under your organization

### Triggering Training

The workflow runs automatically on:
- Push to main branch
- Pull request to main branch
- Manual trigger via GitHub Actions UI

Monitor training progress in the Actions tab of your GitHub repository.
