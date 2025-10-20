# GPU-Accelerated Twitter Sentiment Analysis

This project implements a high-performance, GPU-accelerated machine learning pipeline for binary sentiment analysis. It is trained on the Sentiment140 [dataset](https://www.kaggle.com/datasets/kazanova/sentiment140) (1.6 million tweets) using the NVIDIA RAPIDS ecosystem (`cudf`, `cupy`, and `cuml`).

The primary notebook, `Sentiment Analysis Twitter.ipynb`, demonstrates the entire process from data loading and cleaning to feature extraction and model training, all performed on the GPU.

## Technologies Used

* **NVIDIA RAPIDS:**
    * `cudf`: For GPU-accelerated DataFrames.
    * `cupy`: For GPU-accelerated arrays.
    * `cuml`: For GPU-accelerated machine learning algorithms.
* **Vectorization:** `cuml.feature_extraction.text.TfidfVectorizer`
* **Models:**
    * `cuml.linear_model.LogisticRegression`
    * `cuml.naive_bayes.MultinomialNB`
    * `cuml.neighbors.KNeighborsClassifier`

## Pipeline Overview

1.  **Load Data:** The 1.6M tweet dataset is loaded into a `cudf` DataFrame.
2.  **Preprocess:** Text is cleaned entirely on the GPU (lowercase, remove URLs, mentions `@`, and special characters).
3.  **Vectorize:** The cleaned text is converted into a 50,000-feature TF-IDF matrix using n-grams `(1, 2)`.
4.  **Train & Evaluate:** The data is split (80/20) and used to train and evaluate three different `cuml` models.

## Model Performance

Models were benchmarked on the 20% test split. `LogisticRegression` was the best-performing and most reliable model.

* **Logistic Regression: ~78.4% Accuracy**
    * Showed a well-balanced confusion matrix, correctly identifying both positive and negative classes.

* **Naive Bayes: ~76.8% Accuracy**
    * **Note:** This model failed, exhibiting **mode collapse**. The confusion matrix showed it only predicted one class (negative), making its accuracy score misleading. This is likely an implementation-specific issue with the `cuml` library version used.

* **K-Nearest Neighbors: ~63.2% Accuracy**

## How to Run

This project requires the NVIDIA RAPIDS ecosystem, which runs on a compatible NVIDIA GPU.

1.  **Environment:** Set up a RAPIDS-compatible environment. The easiest way is using the [official Docker container](https://hub.docker.com/r/rapidsai/rapidsai) or a [Conda installation](https://rapids.ai/start.html).
2.  **Dataset:** Download the Sentiment140 dataset and save it in the root folder as `my-dataset.csv`.
3.  **Execute:** Run the `Sentiment Analysis Twitter.ipynb` notebook in your configured environment.
