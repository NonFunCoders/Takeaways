# Takeaways Model

An advanced multilingual AI model built on Mistral 7B, specialized in Bangla language reasoning and understanding with support for question answering, natural language inference, and common sense reasoning.

## Features

- Multi-step reasoning in Bangla with Chain-of-Thought
- Question answering and text comprehension
- Natural language inference and logical reasoning
- Common sense understanding in Bangla
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
- Training Data:
  - Bangla QA Dataset (csebuetnlp/bengali_qa)
  - Bangla Common Sense Dataset (csebuetnlp/bengali_commonsense)
  - Bangla XNLI for Natural Language Inference
- Evaluation:
  - Question answering accuracy
  - Natural language inference performance
  - Common sense reasoning capabilities

## Usage

### Local CLI
```bash
# Question Answering
takeaways "প্রশ্ন: বাংলাদেশের রাজধানী কোথায়?"

# Natural Language Inference
takeaways "বাক্য ১: আকাশ নীল। বাক্য ২: বৃষ্টি হচ্ছে। এই দুই বাক্যের মধ্যে সম্পর্ক কি?"

# Common Sense Reasoning
takeaways "প্রশ্ন: সূর্য কেন পূর্ব দিকে ওঠে?"
```

### API Endpoint
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "প্রশ্ন: বাংলাদেশের রাজধানী কোথায়?"}'
```

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## Automated Training with GitHub Actions

The project includes a GitHub Actions workflow for automated model training and publishing. The workflow:

1. Trains the model using GPU-enabled runners
2. Evaluates Bangla reasoning performance
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

### Evaluation Metrics

The model is evaluated on three main tasks:

1. Bangla Question Answering:
   - Uses the Bengali QA dataset
   - Measures exact match and F1 scores
   - Evaluates reading comprehension ability

2. Natural Language Inference:
   - Uses Bangla XNLI dataset
   - Measures entailment classification accuracy
   - Tests logical reasoning capabilities

3. Common Sense Reasoning:
   - Uses Bengali Common Sense dataset
   - Measures accuracy on everyday reasoning tasks
   - Evaluates real-world knowledge application

### Training Configuration

Model training can be customized through `config/model_config.json`:

```json
{
  "data": {
    "datasets": {
      "bangla_squad": {"enabled": true},
      "bangla_commonsense": {"enabled": true},
      "bangla_xnli": {"enabled": true}
    }
  }
}
```

### Triggering Training

The workflow runs automatically on:
- Push to main branch
- Pull request to main branch
- Manual trigger via GitHub Actions UI

Monitor training progress in the Actions tab of your GitHub repository.

### Additional Language Support

While the model is optimized for Bangla reasoning tasks, it maintains multilingual capabilities from the base Mistral model. To adapt for other languages:

1. Add appropriate datasets in `data/dataset_loader.py`
2. Update evaluation metrics in `evaluation/evaluator.py`
3. Modify prompts and instructions in your target language

## Citations

```bibtex
@misc{bengali_qa,
  title={Bengali Question Answering Dataset},
  author={CSEBUETNLP},
  year={2023}
}

@misc{bengali_commonsense,
  title={Bengali Common Sense Dataset},
  author={CSEBUETNLP},
  year={2023}
}

@misc{xnli,
  title={XNLI: Cross-lingual Natural Language Inference},
  author={Conneau et al.},
  year={2018}
}
