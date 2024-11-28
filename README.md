# Logical Structure Analysis and Question Generation

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Overview
AI-powered system for analyzing logical structures in medical texts and generating TMS preparation questions.

## Installation
```bash
# Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

## Project Structure
.
├── data/                    # Data files and datasets
│   ├── raw/                # Original medical texts
│   └── processed/          # Preprocessed data
│
├── src/                    # Source code
│   ├── data/              
│   │   ├── collector.py    # Data collection utilities
│   │   └── preprocessor.py # Text preprocessing
│   ├── models/            
│   │   ├── gnn.py         # Graph Neural Network
│   │   └── transformer.py  # Question generation model
│   └── utils/             
│       └── helpers.py      # Utility functions
│
├── tests/                  # Unit tests
│   ├── test_preprocessor.py
│   └── test_models.py
│
├── notebooks/              # Jupyter notebooks
├── config/                 # Configuration files
├── .gitignore
├── README.md
├── requirements.txt
├── setup.py
└── .python-version