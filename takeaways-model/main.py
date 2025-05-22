"""
Main script for the Takeaways model.

This script provides a unified interface for the various components
of the Takeaways model, including data processing, training, evaluation,
and deployment.
"""

import os
import sys
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("takeaways.log"),
    ],
)
logger = logging.getLogger(__name__)


def process_data(args):
    """Process datasets for training."""
    logger.info("Processing datasets")
    from scripts.process_datasets import main as process_main
    sys.argv = [sys.argv[0]] + args
    process_main()


def train_model(args):
    """Train the model."""
    logger.info("Training model")
    from model.train import train
    train()


def evaluate_model(args):
    """Evaluate the model."""
    logger.info("Evaluating model")
    from evaluation.evaluate import main as evaluate_main
    evaluate_main()


def serve_local(args):
    """Serve the model locally using Ollama."""
    logger.info("Starting local server")
    from serve.local import main as local_main
    sys.argv = [sys.argv[0]] + args
    local_main()


def serve_api(args):
    """Serve the model via web API."""
    logger.info("Starting API server")
    from serve.api import main as api_main
    sys.argv = [sys.argv[0]] + args
    api_main()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Takeaways: A Superior Coding AI Model")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Data processing command
    process_parser = subparsers.add_parser("process", help="Process datasets")
    process_parser.add_argument("--output", help="Output path for processed data")
    
    # Training command
    train_parser = subparsers.add_parser("train", help="Train the model")
    
    # Evaluation command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the model")
    
    # Local serving command
    local_parser = subparsers.add_parser("serve-local", help="Serve model locally")
    local_parser.add_argument("--port", type=int, help="Port for local server")
    
    # API serving command
    api_parser = subparsers.add_parser("serve-api", help="Serve model via API")
    api_parser.add_argument("--host", help="Host for API server")
    api_parser.add_argument("--port", type=int, help="Port for API server")
    
    args = parser.parse_args()
    
    # Extract arguments for the subcommand
    if args.command == "process":
        process_args = []
        if args.output:
            process_args.extend(["--output", args.output])
        process_data(process_args)
    elif args.command == "train":
        train_model(None)
    elif args.command == "evaluate":
        evaluate_model(None)
    elif args.command == "serve-local":
        local_args = []
        if args.port:
            local_args.extend(["--port", str(args.port)])
        serve_local(local_args)
    elif args.command == "serve-api":
        api_args = []
        if args.host:
            api_args.extend(["--host", args.host])
        if args.port:
            api_args.extend(["--port", str(args.port)])
        serve_api(api_args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
