from setuptools import setup, find_packages

setup(
    name="logical_structure_analysis",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10.0",
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.15.0",
        "torch_geometric>=2.0.0", 
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "wikipedia-api>=0.5.4",
        "pytest>=6.2.5",
        "scipy>=1.7.0",
        "nltk>=3.6.5",
        "pyyaml>=5.4.1",
        "jupyter>=1.0.0",
        "spacy>=3.0.0",
        "networkx>=2.6.0"
    ],
    description="Logical Structure Analysis and Question Generation for TMS Preparation",
    author="JKO, NP, NC",
    license="MIT",
)