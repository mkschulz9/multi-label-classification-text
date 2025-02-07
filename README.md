# Multi-Label Classification in Reddit Comments

> **"Glad I could help. (Side tip: hit return twice to break out of the '>>')"**

**Emotions**: Pride, Relief, Gratitude, Joy

> **"Yawn. They’re toxic together and their only trait seemed to be getting naked together."**

**Emotions**: Fear, Disgust

---

- This project demonstrates a multi-label text classification pipeline using Google's GoEmotions dataset, which comprises Reddit comments labeled with multiple emotions. Each instance can be assigned multiple labels simultaneously in a multi-label classification task.

## Table of Contents

1. [Overview](#overview)
2. [Data Preparation](#data-preparation)
3. [Data Exploration & Visualization](#data-exploration--visualization)
4. [Data Preprocessing](#data-preprocessing)
5. [Feature Extraction & Modeling](#feature-extraction--modeling)
6. [Model Evaluation](#model-evaluation)
7. [Future Work](#future-work)

## Overview

- **Data Source**: [Google's GoEmotions Dataset](https://research.google/blog/goemotions-a-dataset-for-fine-grained-emotion-classification/) (A Dataset for Fine-Grained Emotion Classification)
- **Techniques**: Data Exploration, TF-IDF Feature Extraction, Perceptron, SVM, and Logistic Regression Models, DistilBERT and RoBERTa for Feature Extraction
- **Goal**: Predict what emotions are associated with a text input.

## Data Preparation

### Download Dataset

- The GoEmotions dataset is downloaded and stored locally.
- The file is then read into a Pandas DataFrame, keeping only the required columns (i.e., text and emotion columns).

### Remove Invalid Rows

- Invalid rows are removed (i.e., rows marked as unclear or with no positive emotions labeled).

## Data Exploration & Visualization

- Randomly sample and print a few data samples.
- Calculate and display (via plots):
  - Positive value counts of each emotion label (distribution of emotions).
  - Positive and negative value counts per emotion label.

This step provides an understanding of the dataset’s class distribution and reveals significant class imbalances.

## Data Preprocessing

### Text Cleaning

1. Convert text to lowercase.
2. Expand contractions.
3. Replace URLs with [URL] and @handles with [USER]

### Train-Test Split

- Multi-Label stratified 80/20 split to maintain balanced class proportions.

## Feature Extraction & Modeling

- Features are extracted from the text using TF-IDF, and embeddings are learned from DistilBERT and RoBERTa models. These feature vector representations are then used to train and evaluate classical ML models (e.g., SVM and Logistic Regression models).

### TF-IDF Feature Extraction

- TF-ID is a statistical measure used to evaluate how important a word is within a text sample relative to a collection (corpus) of samples. It is calculated as the product of two metrics: Term Frequency (TF) and Inverse Document Frequency (IDF).
- This simple method effectively highlights words that are frequent in a given document but rare across the corpus, helping distinguish the unique vocabulary of each document. While TF-IDF performs well in many text classification tasks, it does not capture semantical or contextual relationships between words.

### DistilBERT and RoBERTa (Pre-trained Transformer Models) Learned Feature Extraction

- DistilBERT is designed to be lighter and faster while retaining much of BERT's performance. It achieves this by using a distillation process where a smaller model learns to mimic the behavior of a larger one.
- RoBERTa is a robustly optimized BERT approach that modifies key hyperparameters, removes the next-sentence pretraining objective, and trains with much larger mini-batches and learning rates.
- Unlike TD-IDF, these models can capture contextual information and semantic relationships between words, leading to a more feature-dense text representation.

### Model Training

- Once the features are extracted using both methods, they are used to train and evaluate classic ML models. To perform a multi-label classification task, the following approach is adopted:

  - **Binary Relevance**: This is the most straightforward strategy, which treats each label as a separate binary classification problem. Separate binary classifiers are trained for each label, and the final prediction is the union of the predictions of all classifiers.

## Model Evaluation

- After training the models, each model’s performance is evaluated on training and testing data. The following metrics are computed and compared:

### Accuracy

**Definition**: Accuracy measures the proportion of correctly predicted instances (positive and negative) out of the total instances.  
**Formula**:  
![Accuracy](<https://latex.codecogs.com/png.latex?\bg_gray\text{Accuracy}=\frac{\text{True%20Positives%20(TP)}+\text{True%20Negatives%20(TN)}}{\text{Total%20Instances}}>)  
**Use Case**: Useful when the dataset is balanced (similar numbers of positive and negative instances).

### Precision

**Definition**: Precision measures the proportion of correctly predicted positive instances out of all instances predicted as positive (i.e., of all the positive predictions made by the model, what percentage of them are truly positive?).  
**Formula**:  
![Precision](<https://latex.codecogs.com/png.latex?\bg_gray\text{Precision}=\frac{\text{True%20Positives%20(TP)}}{\text{True%20Positives%20(TP)}+\text{False%20Positives%20(FP)}}>)  
**Use Case**: Important in scenarios where minimizing false positives is critical (e.g., spam detection).

### Recall

**Definition**: Recall (or Sensitivity) measures the proportion of correctly predicted positive instances out of all actual positive instances (i.e., out of all the truly positive instances that the model was tested on, what percentage did the model correctly identify as positive?).  
**Formula**:  
![Recall](<https://latex.codecogs.com/png.latex?\bg_gray\text{Recall}=\frac{\text{True%20Positives%20(TP)}}{\text{True%20Positives%20(TP)}+\text{False%20Negatives%20(FN)}}>)  
**Use Case**: Crucial in scenarios where minimizing false negatives is important (e.g., disease detection).

### F1-Score

**Definition**: F1-Score is the harmonic mean of Precision and Recall, providing a single metric that balances both.  
**Formula**:  
![Equation](https://latex.codecogs.com/png.latex?\bg_gray\text{F1-Score}=2\cdot\frac{\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}})  
**Use Case**: Useful when there is an imbalance in class distribution.

### Results

- The table below shows the performance (F1-Score only) of the models on the test set using different feature extraction methods.

| Model               | Feature Extraction Method | F1-Score |
| ------------------- | ------------------------- | -------- |
| Logistic Regression | TF-IDF                    | 13.04    |
| SVM                 | TF-IDF                    | 12.19    |
| Logistic Regression | DistilBERT                | 32.82    |
| SVM                 | DistilBERT                | 0.45     |
| Perceptron          | DistilBERT                | _36.26_  |
| Logistic Regression | RoBERTa                   | 8.83     |
| SVM                 | RoBERTa                   | 17.09    |
| Perceptron          | RoBERTa                   | 13.34    |

- The Perceptron model with DistilBERT features outperforms all other models with an F1-Score of 36.26%. This is up from 13.04% using TF-IDF features.
- **_Note_**: Additional metrics were computed, and other models were evaluated. The top results for brevity are shown above.

## Future Work

- Other classical machine learning models, such as the Perceptron, can be trained using TF-IDF features, and alternative problem formulation methods beyond Binary Relevance are worth exploring.
- Additionally, more straightforward feature extraction techniques like Word2Vec or GloVe can be compared to transformer-based methods to evaluate their performance differences.
- Furthermore, embeddings obtained from DistilBERT and RoBERTa can serve as inputs for more advanced models, such as MLP, LSTM, or GRU, potentially enhancing overall performance.
- Finally, fine-tuning an advanced embedding model on the dataset may lead to better, task-specific representations.
