# Advanced Machine Learning Class Project Qihao Yang 
## IMDb Sentiment Classification
## Overview

This project compares four models for IMDb movie review sentiment classification:

1. TF-IDF + Logistic Regression
2. TF-IDF + Naive Bayes
3. Lightweight CNN
4. DistilBERT

The task is to predict whether a review is positive or negative.

## How to Run

```bash
pip install -r requirements.txt

python src/train_baseline.py
python src/train_cnn.py
python src/train_bert.py
python src/analysis.py
```
## Results

| Model | Test Set | Accuracy | Macro-F1 |
|---|---|---:|---:|
| TF-IDF + Logistic Regression | Full IMDb test set | 0.8804 | 0.8804 |
| TF-IDF + Naive Bayes | Full IMDb test set | 0.8510 | 0.8509 |
| Lightweight CNN | Full IMDb test set | 0.8429 | 0.8424 |
| DistilBERT | 5,000-example subset | 0.8528 | 0.8526 |

The best full-dataset model was TF-IDF + Logistic Regression.

## Files

- `src/train_baseline.py`: trains Logistic Regression and Naive Bayes
- `src/train_cnn.py`: trains the CNN model
- `src/train_bert.py`: trains DistilBERT
- `src/analysis.py`: generates final comparison results and figures

Results are saved in `results/`.  
Figures are saved in `figures/`.

## Note

DistilBERT was trained on a smaller subset for speed.