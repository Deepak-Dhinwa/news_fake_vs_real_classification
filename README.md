# Fake News vs Real News Classification

## Problem Statement

In today's digital age, fake news spreads faster than wildfire. Social media platforms, news aggregators, and even traditional media outlets struggle to distinguish between legitimate journalism and fabricated content. This project tackles the critical challenge of automatically classifying news articles as either **REAL** or **FAKE** using machine learning techniques.

The goal was to build a robust system that can:
- Process large volumes of news text efficiently
- Learn subtle linguistic patterns that distinguish fake from real news
- Provide interpretable predictions for transparency
- Handle the inherent class imbalance in real-world datasets

## Approach

### Data Processing & Preprocessing

**Dataset Composition:**
- **Fake News**: 23,481 articles from various unreliable sources
- **Real News**: 21,417 articles from Reuters (reliable source)
- **Total**: 44,898 articles with title, text, subject, and date

**Key Preprocessing Steps:**
1. **Data Leakage Detection**: Identified that ~29% of articles contained "FAKE" or "REAL" labels in the text, which would artificially inflate model performance
2. **Label Masking**: Replaced leakage words with neutral `[LABEL]` tokens to prevent data leakage
3. **Deduplication**: Removed 5,793 duplicate articles to ensure clean evaluation
4. **Text Combination**: Merged title and text for richer feature representation
5. **Stratified Split**: 80% train, 20% test with preserved class distribution

### Model Architecture & Training

**Traditional ML Pipeline:**
- **CountVectorizer + Logistic Regression**: Baseline approach with 10,000 features
- **TF-IDF + Linear SVM**: Alternative vectorization with calibrated probabilities
- **Feature Engineering**: English stop words removal, max features limitation

**BERT Fine-tuning:**
- **Model**: `bert-base-uncased` with sequence classification head
- **Tokenization**: Fast tokenizer with 256 max length, padding, truncation
- **Training**: 2 epochs, batch size 8, weight decay 0.01
- **Framework**: HuggingFace Transformers with PyTorch backend
- **Monitoring**: Weights & Biases integration for experiment tracking

**Model Interpretability:**
- **LIME Integration**: Local interpretable model explanations for individual predictions
- **Feature Importance**: Text-based explanations showing which words influenced classification

## Results

### Quantitative Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| CountVectorizer + Logistic Regression | 99.0% | 99.0% | 99.0% | 99.0% | 0.999 |
| TF-IDF + Linear SVM | 99.0% | 99.0% | 99.0% | 99.0% | 0.999 |
| BERT Fine-tuned | 100.0% | 100.0% | 100.0% | 100.0% | 1.000 |

### Qualitative Analysis

**Class Distribution:**
- **Fake News**: 52.3% (23,481 articles)
- **Real News**: 47.7% (21,417 articles)
- **Mild imbalance** handled through stratified sampling

**Subject Categories:**
- `politicsNews`: Most common category
- `US_News`: Second most frequent
- `News`: General news articles
- Other specialized categories

**Text Length Analysis:**
- Most articles: 150-1500 words
- Average length: ~500 words
- Long-tail distribution with some very long articles

### Key Insights

1. **Data Quality Matters**: Removing duplicates and handling data leakage was crucial for realistic performance evaluation
2. **BERT Superiority**: Transformer-based models significantly outperformed traditional ML approaches
3. **Feature Engineering**: Simple text combination (title + text) provided rich feature representation
4. **Interpretability**: LIME explanations help understand model decisions for transparency

## Project Structure

```
News Fake vs Real Classification/
├── bert_fake_news_model/          # Trained BERT model & tokenizer
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── vocab.txt
├── results/                       # Training checkpoints
│   ├── checkpoint-500/
│   ├── checkpoint-1000/
│   └── ... (multiple checkpoints)
├── Fake.csv/                      # Fake news dataset
│   └── Fake.csv
├── True.csv/                      # Real news dataset
│   └── True.csv
├── logs/                          # Training logs
├── wandb/                         # Weights & Biases runs
├── fake_news_env/                 # Virtual environment
└── Untitled.ipynb                # Main analysis notebook
```

## Usage Instructions

### Environment Setup
```bash
# Create virtual environment
python -m venv fake_news_env
source fake_news_env/bin/activate  # Linux/Mac
# or
fake_news_env\Scripts\activate     # Windows

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn
pip install transformers torch datasets
pip install lime wandb
```

### Model Loading & Prediction
```python
from transformers import BertTokenizerFast, BertForSequenceClassification
import torch

# Load trained model
model_dir = "./bert_fake_news_model"
tokenizer = BertTokenizerFast.from_pretrained(model_dir)
model = BertForSequenceClassification.from_pretrained(model_dir)
```

## Key Learnings

### Technical Insights
1. **Data Leakage is Real**: Always check for target variable leakage in text data
2. **BERT Dominance**: Transformer models significantly outperform traditional NLP approaches
3. **Deduplication Critical**: Removing duplicates prevents overfitting and ensures fair evaluation
4. **Interpretability Matters**: LIME explanations build trust in model predictions


