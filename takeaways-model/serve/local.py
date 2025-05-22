"""
Local serving script for the Takeaways model using Ollama.

This script allows running the model locally using Ollama, which provides
a fast and efficient way to serve the model on consumer hardware.
"""

import os
import sys
import subprocess
import argparse
import logging
import json

# Add root directory to path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.settings import DEPLOYMENT_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("local_serve.log"),
    ],
)
logger = logging.getLogger(__name__)


def check_ollama_installed():
    """Check if Ollama is installed on the system."""
    try:
        result = subprocess.run(
            ["ollama", "version"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        logger.info(f"Ollama is installed: {result.stdout.strip()}")
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.error("Ollama is not installed or not in PATH")
        logger.info("Please install Ollama from https://ollama.ai")
        return False


def create_model_if_needed():
    """Create the Takeaways model in Ollama if it doesn't exist."""
    try:
        # Check if model exists
        result = subprocess.run(
            ["ollama", "list"], 
            stdout=subprocess.PIPE, 
            text=True,
            check=True
        )
        
        if "takeaways" in result.stdout:
            logger.info("Takeaways model already exists in Ollama")
            return True
        
        # Model doesn't exist, create it
        logger.info("Creating Takeaways model in Ollama")
        
        # Create Modelfile
        modelfile_path = os.path.join(os.path.dirname(__file__), "Modelfile")
        with open(modelfile_path, "w") as f:
            f.write(f"""
FROM mistral:7b
# Set a custom system message that specifies Takeaways behavior
SYSTEM """" 
You are Takeaways, an expert coding assistant specialized in solving complex programming problems 
and explaining code with clarity like a senior engineer. You structure your responses with 
Markdown formatting and provide clear step-by-step explanations before presenting code solutions.

For coding tasks, you should:
1. Break down problems into logical steps
2. Explain your reasoning process
3. Present clean, efficient code solutions
4. Highlight best practices and potential edge cases
5. Use proper formatting with markdown code blocks

You output structured answers with <Explanation>, <CodeBlock>, and <BestPractices> sections.
""""
""")
        
        # Create model
        result = subprocess.run(
            ["ollama", "create", "takeaways", "-f", modelfile_path], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        
        logger.info(f"Model created: {result.stdout.strip()}")
        return True
        
    except subprocess.SubprocessError as e:
        logger.error(f"Error creating model: {e}")
        return False


def run_ollama_server():
    """Run the Ollama server."""
    try:
        logger.info("Starting Ollama server")
        
        # Start the server
        process = subprocess.Popen(
            ["ollama", "serve"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        
        logger.info(f"Ollama server started with PID {process.pid}")
        return process
    except subprocess.SubprocessError as e:
        logger.error(f"Error starting Ollama server: {e}")
        return None


def run_interactive_console():
    """Run an interactive console for the Takeaways model."""
    logger.info("Starting interactive console")
    print("\nTakeaways Coding Assistant")
    print("========================")
    print("Type your coding questions or 'exit' to quit\n")
    
    while True:
        try:
            user_input = input("\n> ")
            
            if user_input.lower() in ["exit", "quit"]:
                break
            
            # Call Ollama API
            result = subprocess.run(
                ["ollama", "run", "takeaways", user_input], 
                stdout=subprocess.PIPE, 
                text=True
            )
            
            print("\n" + result.stdout.strip())
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"\nError: {e}")


def main():
    """Main function to start the local serving."""
    parser = argparse.ArgumentParser(description="Serve Takeaways model locally using Ollama")
    parser.add_argument("--port", type=int, default=DEPLOYMENT_CONFIG["local"]["port"],
                        help="Port for the Ollama server")
    args = parser.parse_args()
    
    # Check if Ollama is installed
    if not check_ollama_installed():
        return
    
    # Create model if needed
    if not create_model_if_needed():
        return
    
    # Start server in the background
    server_process = run_ollama_server()
    
    try:
        # Run interactive console
        run_interactive_console()
    finally:
        # Terminate server when done
        if server_process:
            logger.info("Terminating Ollama server")
            server_process.terminate()
            server_process.wait()


if __name__ == "__main__":
    main()
