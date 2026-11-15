
import re
from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# =====================================================
# PATHS AND SETTINGS
# =====================================================

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "datasets" / "raw" / "WELFake_Dataset.csv"

RANDOM_STATE = 42


# =====================================================
# TEXT CLEANING
# =====================================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =====================================================
# LOAD DATASET
# =====================================================

print("\n========== LOADING DATASET ==========")

df = pd.read_csv(DATA_PATH)

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

df = df.dropna(
    subset=["title", "text", "label"]
).copy()

df["title_clean"] = df["title"].apply(clean_text)
df["text_clean"] = df["text"].apply(clean_text)

df["combined_clean"] = (
    df["title_clean"] + " " + df["text_clean"]
).str.strip()

df = df[df["combined_clean"] != ""].copy()
df["label"] = df["label"].astype(int)

print("Cleaned dataset shape:", df.shape)


# =====================================================
# CREATE ARTICLE GROUPS
# =====================================================

print("\n========== CREATING ARTICLE GROUPS ==========")

# Use article body as the group identifier.
# If body is empty, use the headline instead.

df["group_id"] = df["text_clean"]

empty_body = df["group_id"] == ""

df.loc[empty_body, "group_id"] = (
    df.loc[empty_body, "title_clean"]
)

print("Total rows:", len(df))
print("Unique article groups:", df["group_id"].nunique())


# =====================================================
# IDENTIFY REUTERS-MATCHED GROUPS
# =====================================================

print("\n========== IDENTIFYING REUTERS ARTICLES ==========")

reuters_pattern = re.compile(
    r"\breuters\b",
    flags=re.IGNORECASE
)

df["contains_reuters"] = df["combined_clean"].apply(
    lambda text: bool(reuters_pattern.search(text))
)

reuters_groups = set(
    df.loc[df["contains_reuters"], "group_id"]
)

print("Rows directly mentioning Reuters:",
      int(df["contains_reuters"].sum()))

print("Reuters-matched article groups:",
      len(reuters_groups))


# =====================================================
# GROUP-LEVEL HOLDOUT
# =====================================================

print("\n========== CREATING REUTERS HOLDOUT ==========")

# Every group containing at least one Reuters match
# is reserved for testing.

test_mask = df["group_id"].isin(reuters_groups)

train_df = df[~test_mask].copy()
test_df = df[test_mask].copy()

print("Training samples:", len(train_df))
print("Testing samples:", len(test_df))

if train_df.empty or test_df.empty:
    raise RuntimeError("Training or testing dataset is empty.")

# Verify group leakage

overlap = set(train_df["group_id"]).intersection(
    set(test_df["group_id"])
)

print("Overlapping article groups:", len(overlap))

if overlap:
    raise RuntimeError("Group leakage detected.")

# Verify training set has no Reuters matches

remaining_reuters = train_df["contains_reuters"].sum()

print("Reuters-matched training rows:", remaining_reuters)

if remaining_reuters != 0:
    raise RuntimeError("Reuters records remain in training data.")


# =====================================================
# LABEL DISTRIBUTION
# =====================================================

def show_distribution(data, name):

    print(f"\n{name} label distribution:")

    counts = data["label"].value_counts().sort_index()

    for label in [0, 1]:

        count = int(counts.get(label, 0))
        percentage = count / len(data) * 100

        label_name = "Fake News" if label == 0 else "Real News"

        print(
            f"{label_name}: {count} "
            f"({percentage:.2f}%)"
        )


show_distribution(train_df, "TRAINING")
show_distribution(test_df, "REUTERS TEST")


# =====================================================
# PREPARE DATA
# =====================================================

X_train = train_df["combined_clean"]
y_train = train_df["label"]

X_test = test_df["combined_clean"]
y_test = test_df["label"]


# =====================================================
# MAJORITY CLASS BASELINE
# =====================================================

print("\n========== MAJORITY CLASS BASELINE ==========")

majority_class = y_train.mode().iloc[0]

baseline_predictions = [
    majority_class
] * len(y_test)

print(
    "Majority class:",
    "Fake News" if majority_class == 0 else "Real News"
)

print(
    "Baseline accuracy:",
    f"{accuracy_score(y_test, baseline_predictions) * 100:.2f}%"
)


# =====================================================
# TF-IDF VECTORIZATION
# =====================================================

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


# =====================================================
# MODEL TRAINING
# =====================================================

print("\n========== MODEL TRAINING ==========")

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train_tfidf, y_train)

print("Training completed.")


# =====================================================
# EVALUATION
# =====================================================

print("\n========== REUTERS HOLDOUT EVALUATION ==========")

predictions = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, predictions)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)

print(f"\nAccuracy:          {accuracy * 100:.2f}%")
print(f"Balanced Accuracy: {balanced_accuracy * 100:.2f}%")
print(f"Macro Precision:   {precision * 100:.2f}%")
print(f"Macro Recall:      {recall * 100:.2f}%")
print(f"Macro F1-score:    {f1 * 100:.2f}%")


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Fake News", "Real News"],
        zero_division=0
    )
)


print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )
)


# =====================================================
# FINAL SUMMARY
# =====================================================

print("\n========== FINAL SUMMARY ==========")

print("Training rows:", len(train_df))
print("Reuters test rows:", len(test_df))
print("Reuters groups:", len(reuters_groups))
print("Group overlap:", len(overlap))

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Balanced Accuracy: {balanced_accuracy * 100:.2f}%")
print(f"Macro F1-score: {f1 * 100:.2f}%")

print("\nThis was a diagnostic experiment.")
print("No existing model or vectorizer was overwritten.")

print("\n========== EXPERIMENT COMPLETED ==========")
