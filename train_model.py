from pathlib import Path
import urllib.request
import zipfile
import io
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from preprocessing import clean_text

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "spam_model.joblib"
DATA_PATH = DATA_DIR / "SMSSpamCollection"

UCI_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

def download_dataset():
    DATA_DIR.mkdir(exist_ok=True)
    print("Downloading the public SMS Spam Collection dataset from UCI...")
    with urllib.request.urlopen(UCI_URL, timeout=30) as response:
        archive_bytes = response.read()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        member = next(n for n in archive.namelist() if n.endswith("SMSSpamCollection"))
        DATA_PATH.write_bytes(archive.read(member))
    print(f"Dataset saved to: {DATA_PATH}")

def load_dataset():
    if not DATA_PATH.exists():
        download_dataset()
    df = pd.read_csv(DATA_PATH, sep="\t", header=None, names=["label", "message"], encoding="utf-8")
    df = df.dropna().drop_duplicates()
    df["label"] = df["label"].str.lower().str.strip()
    df = df[df["label"].isin(["ham", "spam"])]
    df["message"] = df["message"].astype(str)
    if len(df) < 100:
        raise ValueError("Dataset is unexpectedly small. Please check data/SMSSpamCollection.")
    return df

def build_pipeline(model):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=clean_text,
            lowercase=False,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            max_features=10000
        )),
        ("classifier", model)
    ])

def train_and_save():
    MODEL_DIR.mkdir(exist_ok=True)
    df = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label"], test_size=0.20, random_state=42, stratify=df["label"]
    )

    candidates = {
        "Multinomial Naive Bayes": build_pipeline(MultinomialNB(alpha=0.1)),
        "Logistic Regression": build_pipeline(LogisticRegression(max_iter=1000, class_weight="balanced"))
    }
    lines = [
        f"Dataset rows after cleaning: {len(df)}",
        f"Training messages: {len(X_train)}",
        f"Test messages: {len(X_test)}",
        ""
    ]
    best_name, best_model, best_f1 = None, None, -1
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, pos_label="spam", zero_division=0)
        recall = recall_score(y_test, predictions, pos_label="spam", zero_division=0)
        f1 = f1_score(y_test, predictions, pos_label="spam", zero_division=0)
        lines += [
            name,
            f"Accuracy:  {accuracy:.4f}",
            f"Precision: {precision:.4f}",
            f"Recall:    {recall:.4f}",
            f"F1-score:  {f1:.4f}",
            classification_report(y_test, predictions, zero_division=0),
            ""
        ]
        if f1 > best_f1:
            best_name, best_model, best_f1 = name, model, f1

    joblib.dump(best_model, MODEL_PATH)
    lines.insert(0, f"Selected model (highest spam F1 on held-out split): {best_name}")
    (MODEL_DIR / "metrics.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"Saved model to {MODEL_PATH}")
    return best_model

if __name__ == "__main__":
    train_and_save()
