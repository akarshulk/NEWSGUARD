
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parent

model = joblib.load(ROOT / "models" / "fake_news_model.pkl")
vectorizer = joblib.load(ROOT / "models" / "tfidf_vectorizer.pkl")

feature_names = vectorizer.get_feature_names_out()
classifier = model

print("\n========== MODEL DIAGNOSTICS ==========")

# Logistic Regression coefficients
coefficients = classifier.coef_[0]

# Positive coefficients favour class 1 (Real)
top_real = coefficients.argsort()[-25:][::-1]

# Negative coefficients favour class 0 (Fake)
top_fake = coefficients.argsort()[:25]

print("\nTOP WORDS ASSOCIATED WITH REAL NEWS:")

for index in top_real:
    print(feature_names[index], round(coefficients[index], 4))

print("\nTOP WORDS ASSOCIATED WITH FAKE NEWS:")

for index in top_fake:
    print(feature_names[index], round(coefficients[index], 4))

print("\n========== COMPLETE ==========")
