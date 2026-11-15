
import re
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "fake_news_model.pkl"
VECTORIZER_PATH = ROOT / "models" / "tfidf_vectorizer.pkl"

# Load saved model and vectorizer
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_news(title, article):
    content = clean_text(title + " " + article)

    if not content:
        return None

    features = vectorizer.transform([content])

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    confidence = max(probabilities) * 100

    if prediction == 0:
        result = "Fake News"
    else:
        result = "Real News"

    return {
        "prediction": result,
        "confidence": round(confidence, 2)
    }


# Test the prediction module
if __name__ == "__main__":
    title = input("Enter news headline: ")
    article = input("Enter news article: ")

    result = predict_news(title, article)

    if result:
        print("\n========== NEWSGUARD AI ==========")
        print("Prediction:", result["prediction"])
        print("Confidence:", str(result["confidence"]) + "%")
    else:
        print("Please enter valid news content.")
