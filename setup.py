#!/usr/bin/env python
from setuptools import setup, find_packages

setup(name="synthax",
      version="0.1.0",
      url="https://github.com/PapayaResearch/synthax",
      author="Manuel Cherep, Nikhil Singh",
      author_email="mcherep@mit.edu, nsingh1@mit.edu",
      description="SynthAX: A Fast Modular Synthesizer in JAX",
      long_description=open("README.md").read(),
      long_description_content_type="text/markdown",
      packages=find_packages(),
      python_requires=">=3.7",
      install_requires=["jax>=0.3.0",
                        "jaxlib>=0.3.0",
                        "chex",
                        "flax",
                        "PyYAML"
                        ])
