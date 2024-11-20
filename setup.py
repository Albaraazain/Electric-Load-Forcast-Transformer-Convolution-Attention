# setup.py
from setuptools import setup, find_packages

setup(
    name="transformer_conv_attention",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        line.strip()
        for line in open("requirements.txt").readlines()
        if not line.startswith("#")
    ],
    author="Your Name",
    description="Transformer model with convolutional attention for time series",
    python_requires=">=3.8",
)