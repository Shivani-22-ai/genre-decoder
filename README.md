# Task 1: Movie Genre Classification

**CODSOFT Machine Learning Internship**

## Problem Statement
Predict a movie's genre based on its plot summary, using text classification techniques.

## Dataset
[Genre Classification Dataset IMDb (Kaggle)](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb)
- `train_data.txt` — records in the format `ID ::: TITLE ::: GENRE ::: DESCRIPTION`
- `test_data.txt` / `test_data_solution.txt` — official test set and its true labels (not required by this notebook, which evaluates using its own train/test split)

Not included in this repo due to file size and repo hygiene. Download from the link above and place `train_data.txt` in `data/`.

## Approach
1. **Data loading** — parsed the `:::`-delimited text file into a structured table
2. **EDA** — examined genre distribution and plot description length
3. **Text preprocessing** — lowercased text and removed non-alphabetic characters
4. **Feature extraction** — TF-IDF vectorization (unigrams + bigrams, 20,000 max features, English stop words removed)
5. **Models compared** — Multinomial Naive Bayes, Logistic Regression, Linear SVM (all with `class_weight="balanced"` where applicable, to counter genre imbalance)
6. **Evaluation** — Accuracy, Macro-F1, and Weighted-F1 (accuracy alone is misleading given how imbalanced the genre distribution is — Drama and Documentary dominate the dataset)

## Results
| Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|
| Multinomial Naive Bayes | 0.500 | 0.099 | 0.401 |
| Logistic Regression | 0.488 | 0.365 | 0.512 |
| Linear SVM | 0.529 | 0.348 | 0.536 |


## How to Run
```bash
cd task1_movie_genre_classification
jupyter notebook notebooks/movie_genre_classification.ipynb
```

## Tech Stack
Python, pandas, scikit-learn, TF-IDF, matplotlib, seaborn

---
*Part of the [CODSOFT Machine Learning Internship](../README.md) task series.*