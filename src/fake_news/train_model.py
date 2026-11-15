
import re
import shutil
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold


# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "datasets" / "raw" / "WELFake_Dataset.csv"
MODEL_DIR = ROOT / "models"

MODEL_PATH = MODEL_DIR / "fake_news_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# TEXT CLEANING
# --------------------------------------------------

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

print("\n========== LOADING DATASET ==========")

df = pd.read_csv(DATA_PATH)

print("Original shape:", df.shape)

if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

df.dropna(subset=["title", "text", "label"], inplace=True)

df["title_clean"] = df["title"].apply(clean_text)
df["text_clean"] = df["text"].apply(clean_text)

df["combined_clean"] = (
    df["title_clean"] + " " + df["text_clean"]
).str.strip()

df = df[df["combined_clean"] != ""].copy()

df["label"] = df["label"].astype(int)

print("Cleaned shape:", df.shape)

print("\nLabel distribution:")
print(df["label"].value_counts().sort_index())


# --------------------------------------------------
# CREATE GROUPS TO PREVENT DATA LEAKAGE
# --------------------------------------------------

print("\n========== CREATING GROUPS ==========")

# Articles with identical cleaned bodies belong to
# the same group. If the body is empty, use the title.
df["group_id"] = df["text_clean"]

empty_body = df["group_id"] == ""

df.loc[empty_body, "group_id"] = (
    df.loc[empty_body, "title_clean"]
)

# This prevents identical article bodies from being
# distributed across both training and testing sets.

print("Total rows:", len(df))
print("Unique article groups:", df["group_id"].nunique())
print(
    "Rows belonging to repeated article groups:",
    df.duplicated(subset=["group_id"], keep=False).sum()
)


# --------------------------------------------------
# STRATIFIED GROUP TRAIN-TEST SPLIT
# --------------------------------------------------

print("\n========== GROUP-AWARE TRAIN-TEST SPLIT ==========")

X = df["combined_clean"]
y = df["label"]
groups = df["group_id"]

splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

train_indices, test_indices = next(
    splitter.split(X, y, groups)
)

X_train = X.iloc[train_indices]
X_test = X.iloc[test_indices]

y_train = y.iloc[train_indices]
y_test = y.iloc[test_indices]

train_groups = groups.iloc[train_indices]
test_groups = groups.iloc[test_indices]

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining label distribution:")
print(y_train.value_counts().sort_index())

print("\nTesting label distribution:")
print(y_test.value_counts().sort_index())


# --------------------------------------------------
# VERIFY THAT GROUPS DO NOT OVERLAP
# --------------------------------------------------

print("\n========== LEAKAGE VERIFICATION ==========")

group_overlap = set(train_groups).intersection(set(test_groups))

print("Overlapping article groups:", len(group_overlap))

# Additional checks for exact text overlap.
body_overlap = set(
    df.iloc[train_indices]["text_clean"]
).intersection(
    set(df.iloc[test_indices]["text_clean"])
)

combined_overlap = set(
    X_train
).intersection(set(X_test))

print("Exact article-body overlap:", len(body_overlap))
print("Exact combined-text overlap:", len(combined_overlap))

if group_overlap:
    raise RuntimeError(
        "Data leakage detected: groups overlap between train and test."
    )

print("Leakage verification passed.")


# --------------------------------------------------
# TF-IDF VECTORIZATION
# --------------------------------------------------

print("\n========== TF-IDF VECTORIZATION ==========")

vectorizer = TfidfVectorizer(
    max_features=50000,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("Training TF-IDF shape:", X_train_tfidf.shape)
print("Testing TF-IDF shape:", X_test_tfidf.shape)


# --------------------------------------------------
# TRAIN LOGISTIC REGRESSION
# --------------------------------------------------

print("\n========== MODEL TRAINING ==========")

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train_tfidf, y_train)

print("Model training completed.")


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

print("\n========== MODEL EVALUATION ==========")

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, average="weighted", zero_division=0
)

recall = recall_score(
    y_test, y_pred, average="weighted", zero_division=0
)

f1 = f1_score(
    y_test, y_pred, average="weighted", zero_division=0
)

print(f"\nAccuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
print(f"F1-score:  {f1 * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Fake News", "Real News"],
        zero_division=0
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))


# --------------------------------------------------
# BACK UP EXISTING MODEL FILES
# --------------------------------------------------

print("\n========== BACKING UP EXISTING MODELS ==========")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

for file_path in [MODEL_PATH, VECTORIZER_PATH]:
    if file_path.exists():
        backup_path = file_path.with_name(
            f"{file_path.stem}_backup_{timestamp}{file_path.suffix}"
        )

        shutil.copy2(file_path, backup_path)
        print("Backup created:", backup_path.name)


# --------------------------------------------------
# SAVE UPDATED MODEL AND VECTORIZER
# --------------------------------------------------

print("\n========== SAVING UPDATED MODEL ==========")

joblib.dump(model, MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)

print("Model saved:", MODEL_PATH)
print("Vectorizer saved:", VECTORIZER_PATH)

print("\n========== TRAINING PIPELINE COMPLETED ==========")
