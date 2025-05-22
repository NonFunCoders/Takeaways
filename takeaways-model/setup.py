"""
Setup script for the Takeaways model package.
"""

from setuptools import setup, find_packages

setup(
    name="takeaways-model",
    version="0.1.0",
    description="A specialized AI model for coding tasks",
    author="Takeaways Team",
    author_email="example@example.com",
    packages=find_packages(),
    install_requires=[
        "transformers>=4.36.0",
        "torch>=2.0.0",
        "peft>=0.7.0",
        "accelerate>=0.25.0",
        "bitsandbytes>=0.41.0",
        "datasets>=2.14.0",
        "evaluate>=0.4.0",
        "sentencepiece>=0.1.99",
        "protobuf>=4.24.4",
        "fastapi>=0.104.1",
        "uvicorn>=0.24.0",
        "websockets>=11.0.3",
        "ollama>=0.1.6",
        "python-dotenv>=1.0.0",
        "numpy>=1.24.0",
        "pandas>=2.1.0",
        "tqdm>=4.66.1",
    ],
    python_requires=">=3.8",
)
