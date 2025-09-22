# 📧 Spam Classifier – DA5400

This repository contains a spam classifier built from scratch as part of **DA5400: Foundation in Machine Learning** at **IIT Madras**.

## 👤 Author

- **Name**: Vyas Vedanshu Gaurangbhai  
- **Roll Number**: ME21B218

## 🎯 Objective

To develop a spam classifier using labeled email data and machine learning techniques, with a focus on feature extraction, model training, and evaluation.

## 📁 Dataset

- **Source**: Enron Email Dataset  
- **Training**: 2000 spam + 2000 ham emails  
- **Validation**: 200 spam + 200 ham emails  
- Additional emails added to improve relevance to Indian context.

## 🔍 Feature Extraction

- Word frequency used as features.
- Stop words removed.
- Words with frequency > 3000 filtered out.
- Feature extraction done using `CountVectorizer`.

## 🧠 Model Architecture

- **Classifier Used**: Support Vector Machine (SVM)
- **Training Script**: `Train.py`
- **Classification Script**: `Classify.py`
- **Saved Files**: `model.object`, `count_vectorizer.object`

## ⚙️ Algorithm Steps

1. Load and preprocess emails (lowercase, remove stop words).
2. Extract features using `CountVectorizer`.
3. Train SVM with:
   - Linear kernel
   - RBF kernel (C = 1, 2, 3)
   - Polynomial kernel (degree = 2, 3, 4)
4. Select best model based on validation accuracy.

## 📊 Cross-Validation Results

| Kernel                     | Accuracy (%) |
|---------------------------|--------------|
| Linear                    | 95.26        |
| RBF (C=1)                 | 95.51        |
| RBF (C=2) ✅ Best         | 96.76        |
| RBF (C=3)                 | 96.26        |
| Polynomial (degree=2)     | 76.06        |
| Polynomial (degree=3)     | 61.60        |
| Polynomial (degree=4)     | 57.11        |

## 📈 Final Results

- Run `Classify.py` to classify test emails.
- Ensure `model.object` and `count_vectorizer.object` are in the same directory.
- Output saved to `output.txt` and printed to console.

---
