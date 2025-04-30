#  Detecting Sentiment and Potential Bias in Amazon Reviews

## Problem Statement

In today’s society, businesses heavily rely on user feedback for optimization and improvement of products and services. Understanding public sentiment through reviews and feedback is essential for businesses to make decisions. The issue is that manually analyzing reviews is a tedious process due to the volume of data. This project automates sentiment detection using machine learning and NLP techniques to classify Amazon Fine Food Reviews as positive, neutral, or negative. It also compares model performance on structured vs. unstructured data and investigates potential sentiment bias in automated classifiers.

---

## Dataset

We use the [Amazon Fine Food Reviews dataset](http://kaggle.com/datasets/snap/amazon-fine-food-reviews), which contains over 500,000 reviews including text, ratings, and metadata. In this project, a subset is used for benchmarking and exploratory analysis.

We also use the [Twitter SST dataset](https://www.kaggle.com/datasets/kazanova/sentiment140) 

---

## Clone the Repository


```bash
git clone https://github.com/farzadashfak/nlp_project.git
cd nlp_project
```

---

## Installation

To install all necessary dependencies, run the following:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch transformers datasets nltk tqdm evaluate scipy
```

Additionally, ensure that NLTK resources are downloaded:
```python
import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')
```

---

## Notebooks Overview

### `benchmark_sentiment_analysis_new.ipynb`

- **Purpose**: Compare different sentiment classification models on a combined benchmark dataset (Amazon + SST).

### `exploratory_data_analysis.ipynb`

- **Purpose**: Explore patterns in Amazon and Twitter reviews prior to modeling.

---

## Models Compared

- VADER
- TF-IDF + Logistic Regression
- Pretrained RoBERTa
- Fine-tuned RoBERTa
