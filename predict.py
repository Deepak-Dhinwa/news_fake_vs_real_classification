"""Classify one article with the saved BERT model. Does not train."""
import sys
from pathlib import Path

import torch
from transformers import BertForSequenceClassification, BertTokenizerFast

MODEL_DIR = Path(__file__).resolve().parent / "bert_fake_news_model"
LABELS = {0: "REAL", 1: "FAKE"}


def classify(text: str) -> str:
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        pred = int(torch.argmax(model(**inputs).logits, dim=-1).item())
    return LABELS[pred]


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]).strip() or (
        "The central bank raised interest rates on Wednesday after officials met in the capital."
    )
    print(classify(text))
