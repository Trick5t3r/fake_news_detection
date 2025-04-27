# Fake News Detection System

A machine learning-based system for detecting fake news articles using BERT-based text classification.

## Project Overview

This project aims to combat the spread of misinformation by automatically detecting fake news articles. The system uses BERT-based text classification to analyze news content and determine its veracity. The model is trained on a dataset of labeled news articles from Kaggle, learning to distinguish between reliable and unreliable sources.

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Kaggle 

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fake_news_detection.git
cd fake_news_detection
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Download the dataset:
   - Download from Kaggle: https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets/data
   - Place `True.csv` and `Fake.csv` in the `data/` directory

## Project Structure

```
fake_news_detection/
├── data/                    # Dataset storage
│   ├── True.csv           # Real news articles
│   └── Fake.csv           # Fake news articles
├── models/                 # Model definitions and weights
│   ├── bert.py           # BERT model implementation
│   └── models_trained/   # Trained model weights
├── src/                    # Source code
│   ├── train_model.py    # Model training script
│   └── fakenewsdetection.py # Detection and evaluation script
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## Usage

### Training the Model

To train the model:
```bash
python src/train_model.py
```

This will:
- Load and preprocess the dataset
- Train the BERT-based classifier
- Save the trained model to `models/models_trained/bert_fake_news_classifier.pt`

### Using the Detection System

1. For interactive text prediction:
```bash
python src/fakenewsdetection.py
```

2. For model evaluation:
```bash
python src/fakenewsdetection.py --evaluate
```

## Model Architecture

The system uses a BERT-based architecture for text classification:

1. Text Preprocessing:
   - Tokenization using BERT tokenizer
   - Maximum sequence length: 256 tokens
   - Special token addition ([CLS], [SEP])
   - Padding and truncation

2. Model Architecture:
   - BERT base model (bert-base-uncased)
   - Dropout layer (p=0.3)
   - Classification head
   - Binary classification output (Fake/Real)

3. Training Configuration:
   - Batch size: 16
   - Learning rate: 2e-5
   - Epochs: 3
   - Validation split: 10%

### Model Performance

The model's performance is evaluated using standard classification metrics:

- Accuracy: 1
- F1-Score: 1
- Precision: 1
- Recall: 1
