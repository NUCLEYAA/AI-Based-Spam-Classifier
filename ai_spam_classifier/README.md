# AI-Based Spam Classifier

A Flask web application that classifies SMS/email-style text as **SPAM** or **NOT SPAM** using NLP and machine learning.

## Features
- Message classification with a confidence estimate
- Text normalization and TF-IDF features (unigrams + bigrams)
- Compares Multinomial Naive Bayes and Logistic Regression
- Selects the model with the higher spam-class F1 score on a held-out test split
- Stores prediction history in SQLite
- Model evaluation page

## Requirements
- Python 3.10 or newer
- Internet connection for the first dataset download

## Setup (Windows)
1. Extract the ZIP.
2. Open the extracted folder in VS Code.
3. Open Terminal and create a virtual environment:

   ```bash
   py -m venv venv
   venv\Scripts\activate
   ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Train the model (this downloads the public UCI SMS Spam Collection dataset on first run):

   ```bash
   python train_model.py
   ```

6. Start the website:

   ```bash
   python app.py
   ```

7. Open `http://127.0.0.1:5000` in your browser.

## Project structure

```text
ai_spam_classifier/
├── app.py
├── train_model.py
├── preprocessing.py
├── database.py
├── requirements.txt
├── README.md
├── data/       # downloaded dataset is stored here
├── models/     # trained model and evaluation metrics
├── templates/  # Flask HTML pages
└── static/     # CSS
```

## How it works
1. Load labelled messages (`ham` or `spam`).
2. Remove missing/duplicate rows.
3. Normalize text (lowercase, URL/number tokens, punctuation cleanup).
4. Convert text into TF-IDF features, including one-word and two-word patterns.
5. Train and compare Multinomial Naive Bayes and Logistic Regression.
6. Evaluate using accuracy, precision, recall and F1-score.
7. Save the selected pipeline with joblib.
8. Flask loads the pipeline to classify submitted messages.
9. Save each prediction in SQLite.

## Important notes
- The confidence shown is the model's estimated class probability, not a guarantee of safety.
- This academic prototype analyses message text only. It does not inspect sender identity, email headers, attachments or URLs.
- Do not use this prototype as the only protection against phishing or fraud.
- Do not claim 100% real-world accuracy. Report the metrics produced by your own run and explain the test split.
- The first run needs internet to download the UCI dataset. The trained app can run offline after the dataset and model have been downloaded/generated.
