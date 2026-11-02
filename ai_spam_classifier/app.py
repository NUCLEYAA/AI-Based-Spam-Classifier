from flask import Flask, render_template, request, redirect, url_for, flash
from pathlib import Path
import joblib
from database import init_db, save_prediction, get_history, clear_history
from train_model import train_and_save, MODEL_PATH

app = Flask(__name__)
app.secret_key = "change-this-secret-key"
init_db()

def get_model():
    if not MODEL_PATH.exists():
        train_and_save()
    return joblib.load(MODEL_PATH)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    message = ""
    if request.method == "POST":
        message = request.form.get("message", "").strip()
        if not message:
            flash("Please enter a message before checking it.", "error")
        elif len(message) > 5000:
            flash("Message is too long. Please keep it under 5,000 characters.", "error")
        else:
            model = get_model()
            prediction = model.predict([message])[0]
            probabilities = model.predict_proba([message])[0]
            classes = list(model.classes_)
            spam_index = classes.index("spam")
            confidence = float(probabilities[spam_index] if prediction == "spam"
                               else probabilities[classes.index("ham")])
            result = {
                "label": "SPAM" if prediction == "spam" else "NOT SPAM",
                "is_spam": prediction == "spam",
                "confidence": round(confidence * 100, 2)
            }
            save_prediction(message, result["label"], result["confidence"])
    return render_template("index.html", result=result, message=message)

@app.route("/history")
def history():
    return render_template("history.html", records=get_history())

@app.route("/history/clear", methods=["POST"])
def history_clear():
    clear_history()
    flash("Prediction history cleared.", "success")
    return redirect(url_for("history"))

@app.route("/about")
def about():
    metrics_path = Path("models/metrics.txt")
    metrics = metrics_path.read_text(encoding="utf-8") if metrics_path.exists() else "Model metrics are not available yet. Train the model first."
    return render_template("about.html", metrics=metrics)

if __name__ == "__main__":
    app.run(debug=True)
