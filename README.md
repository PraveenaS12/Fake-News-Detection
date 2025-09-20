# Fake News Detection using NLP and Machine Learning

![Fake News Banner](https://user-images.githubusercontent.com/79269520/150174169-1ddc4349-3e3a-4395-8167-154a43b952a2.png)

## 📖 Overview

This project is an end-to-end machine learning solution for **detecting fake news**. Given a news article's title and text, the model can classify it as either **"Real"** or **"Fake"** with high accuracy. The project leverages Natural Language Processing (NLP) for text preprocessing and a Logistic Regression classifier to make the final predictions.

Trained on a large, real-world dataset, the model achieves an accuracy of over **98%**, making it a powerful tool in the fight against misinformation. This repository includes the full training pipeline in a Jupyter Notebook, as well as the final pre-trained model and vectorizer for immediate use.

---

## 📂 Repository Contents

This repository is structured to provide both a complete walkthrough of the project and a ready-to-use prediction tool.

1.  **`Fake News Detection.ipynb`**: The main Jupyter Notebook containing the entire workflow. This includes data loading and cleaning, exploratory data analysis, text preprocessing, feature extraction (TF-IDF), model training, and performance evaluation.
2.  **`fake_news_model.pkl`**: The final, pre-trained Logistic Regression model, saved using `joblib`. This can be loaded to make instant predictions on new data.
3.  **`tfidf_vectorizer.pkl`**: The fitted `TfidfVectorizer` object. This is essential for transforming new text data into the exact numerical format that the trained model requires.
4.  **`README.md`**: This file, providing a complete guide to the project.

---

## 💾 Dataset

This project utilizes the **"Fake and Real News Dataset"** from Kaggle, a comprehensive collection of news articles labeled by human fact-checkers.

*   **Link:** [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

**Instructions:**
To run the full `Fake News Detection.ipynb` notebook from scratch, you must download the dataset from the link above. After unzipping, place the `Fake.csv` and `True.csv` files in the same root directory as the notebook.

---

## ✨ Project Workflow & Features

The project follows a systematic machine learning pipeline:

1.  **Data Loading & Merging:** The `Fake.csv` and `True.csv` files are loaded and merged into a single, shuffled DataFrame.
2.  **Text Preprocessing:** A custom pipeline cleans the article text by:
    *   Converting all text to lowercase.
    *   Removing stopwords (e.g., "the", "a", "in").
    *   Applying lemmatization to reduce words to their root form (e.g., "studies" -> "study").
    *   Stripping out punctuation, URLs, and numbers.
3.  **Feature Extraction with TF-IDF:** The cleaned text is vectorized using **TF-IDF**, which converts text into numerical features that represent the importance of words in the articles.
4.  **Model Training:** A **Logistic Regression** model is trained on the TF-IDF features to learn the patterns that differentiate fake news from real news.
5.  **Performance Evaluation:** The model's predictive power is rigorously assessed on an unseen test set using metrics like **Accuracy**, **F1-Score**, a detailed **Classification Report**, and a **Confusion Matrix**.
6.  **Bonus Visualization:** **Word Clouds** are generated to visually highlight the most frequent and indicative words in both fake and real news articles.

---

## 🛠️ Tech Stack & Libraries

*   **Language:** Python
*   **Core Libraries:**
    *   **Pandas & NumPy:** For data handling and numerical operations.
    *   **NLTK:** For the complete text preprocessing pipeline.
    *   **Scikit-learn:** For data splitting, TF-IDF vectorization, model training (Logistic Regression), and evaluation metrics.
    *   **Joblib:** For saving and loading the pre-trained model and vectorizer.
    *   **Matplotlib & Seaborn:** For creating the confusion matrix and other plots.
    *   **WordCloud:** For the bonus visualization task.

---

## 🚀 How to Use the Pre-trained Model

To get instant predictions without retraining the model, you can load the `.pkl` files.

1.  Ensure you have `scikit-learn` and `nltk` installed.
2.  Use the following code snippet to classify any new text:

**Example Code:**
```
import joblib

# Load the saved vectorizer and model
vectorizer = joblib.load('tfidf_vectorizer.pkl')
model = joblib.load('fake_news_model.pkl')

# Example of a new news article title + text
new_article = "Scientists discover a new planet in our solar system made entirely of diamond."

# Note: You would need a preprocessing function here similar to the one in the notebook.
# For simplicity, this example skips that step, but it's crucial for real use.
# preprocessed_article = preprocess_text(new_article)

# 1. Transform the new article using the loaded vectorizer
new_article_tfidf = vectorizer.transform([preprocessed_article])

# 2. Make a prediction
prediction = model.predict(new_article_tfidf)

# 3. Interpret the result (0 = Fake, 1 = Real)
if prediction == 1:
    print("The article is predicted to be REAL.")
else:
    print("The article is predicted to be FAKE.")
```

---

## ⚙️ How to Run the Full Project

To run the entire training and evaluation pipeline yourself:

1.  **Clone the repository:**
    ```
    git clone https://github.com/your-username/your-repo-name.git
    ```
2.  **Install dependencies** from a `requirements.txt` file.
    ```
    pip install -r requirements.txt
    ```
3.  **Download NLTK resources** (stopwords, punkt, wordnet).
4.  **Download the dataset** from the Kaggle link and place `Fake.csv` and `True.csv` in the root folder.
5.  **Launch Jupyter Notebook** and open `Fake News Detection.ipynb`.

---

## 📊 Results & Evaluation

The Logistic Regression model achieved outstanding performance on the unseen test set.

*   **Final Model Accuracy:** **98.76%**
*   **Final Model F1-Score:** **0.9871**
```
