

logical_structure_analysis/
├── data/
│   ├── raw/                    # Raw Wikipedia articles
│   └── processed/              # Preprocessed text data
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_collector.py   # Wikipedia API integration
│   │   ├── preprocessor.py     # Text cleaning and preprocessing
│   │   └── dataset.py         # PyTorch dataset classes
│   ├── models/
│   │   ├── __init__.py
│   │   ├── gnn.py             # GNN model implementation
│   │   ├── transformer.py      # Transformer model implementation
│   │   └── embeddings.py      # Text embedding utilities
│   ├── question_generation/
│   │   ├── __init__.py
│   │   ├── rule_engine.py     # Rule-based question generation
│   │   └── templates.py       # Question templates
│   └── utils/
│       ├── __init__.py
│       └── metrics.py         # Evaluation metrics
├── tests/
│   ├── __init__.py
│   ├── test_data_collector.py
│   ├── test_preprocessor.py
│   └── test_models.py
├── notebooks/
│   ├── data_exploration.ipynb
│   └── model_evaluation.ipynb
├── config/
│   └── config.yaml            # Configuration parameters
├── requirements.txt
├── setup.py
└── README.md