
import re
from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedGroupKFold


# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "datasets" / "raw" / "WELFake_Dataset.csv"


# --------------------------------------------------
# PUBLISHER NAMES TO MASK
# --------------------------------------------------

PUBLISHERS = [
    "Associated Press",
    "New York Times",
    "Washington Post",
    "Breitbart",
    "Reuters",
    "CNN",
    "BBC",
    "Fox News",
    "Guardian",
]

publisher_pattern = re.compile(
    r"\b(?:" + "|".join(
        re.escape(name)
        for name in sorted(PUBLISHERS, key=len, reverse=True)
    ) + r")\b",
    flags=re.IGNORECASE
)


# --------------------------------------------------
# TEXT CLEANING
# --------------------------------------------------

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def mask_publishers(text):
    return publisher_pattern.sub(" ", text)


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

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

print("Dataset shape:", df.shape)


# --------------------------------------------------
# PUBLISHER-LABEL ASSOCIATION INSPECTION
# --------------------------------------------------

print("\n========== PUBLISHER LABEL DISTRIBUTION ==========")

for publisher in PUBLISHERS:

    pattern = re.compile(
        r"\b" + re.escape(publisher) + r"\b",
        flags=re.IGNORECASE
    )

    mask = df["combined_clean"].apply(
        lambda x: bool(pattern.search(x))
    )

    subset = df[mask]

    if len(subset) == 0:
        continue

    fake_count = (subset["label"] == 0).sum()
    real_count = (subset["label"] == 1).sum()

    fake_percentage = fake_count / len(subset) * 100

    print(f"\nPublisher: {publisher}")
    print("Total matches:", len(subset))
    print("Fake-labelled:", fake_count)
    print("Real-labelled:", real_count)
    print(f"Fake-labelled percentage: {fake_percentage:.2f}%")


# --------------------------------------------------
# CREATE GROUPS
# --------------------------------------------------

print("\n========== GROUP-AWARE SPLIT ==========")

df["group_id"] = df["text_clean"]

empty_body = df["group_id"] == ""

df.loc[empty_body, "group_id"] = (
    df.loc[empty_body, "title_clean"]
)

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

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

group_overlap = set(
    groups.iloc[train_indices]
).intersection(
    set(groups.iloc[test_indices])
)

print("Overlapping article groups:", len(group_overlap))

if group_overlap:
    raise RuntimeError("Group leakage detected.")


# --------------------------------------------------
# EVALUATION FUNCTION
# --------------------------------------------------

def evaluate_model(X_train_data, X_test_data, name):

    print(f"\n========== {name} ==========")

    vectorizer = TfidfVectorizer(
        max_features=50000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2
    )

    train_features = vectorizer.fit_transform(
        X_train_data
    )

    test_features = vectorizer.transform(
        X_test_data
    )

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(train_features, y_train)

    predictions = model.predict(test_features)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    print(f"\nAccuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1-score:  {f1 * 100:.2f}%")

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

    return {
        "Accuracy": round(accuracy * 100, 2),
        "Precision": round(precision * 100, 2),
        "Recall": round(recall * 100, 2),
        "F1-score": round(f1 * 100, 2),
    }


# --------------------------------------------------
# EXPERIMENT 1: ORIGINAL TEXT
# --------------------------------------------------

baseline_results = evaluate_model(
    X_train,
    X_test,
    "BASELINE MODEL - ORIGINAL TEXT"
)


# --------------------------------------------------
# EXPERIMENT 2: PUBLISHER-MASKED TEXT
# --------------------------------------------------

print("\n========== MASKING PUBLISHER NAMES ==========")

X_train_masked = X_train.apply(mask_publishers)
X_test_masked = X_test.apply(mask_publishers)

print("Publisher names masked in training and testing text.")


masked_results = evaluate_model(
    X_train_masked,
    X_test_masked,
    "PUBLISHER-MASKED MODEL"
)


# --------------------------------------------------
# FINAL COMPARISON
# --------------------------------------------------

print("\n========== FINAL COMPARISON ==========")

comparison = pd.DataFrame(
    [baseline_results, masked_results],
    index=["Baseline", "Publisher-Masked"]
)

print(comparison)

accuracy_drop = (
    baseline_results["Accuracy"]
    - masked_results["Accuracy"]
)

print(
    f"\nAccuracy change after masking: "
    f"{accuracy_drop:.2f} percentage points"
)

if accuracy_drop > 3:
    print(
        "\nNOTICE: Performance dropped noticeably after "
        "publisher masking. Publisher-related text may "
        "be contributing to the model's predictions."
    )

elif accuracy_drop < -3:
    print(
        "\nNOTICE: Performance improved after publisher "
        "masking. Publisher-related text may have been "
        "introducing noise."
    )

else:
    print(
        "\nNOTICE: Performance changed only slightly. "
        "This experiment does not show a large dependence "
        "on the selected publisher names."
    )

print("\n========== EXPERIMENT COMPLETED ==========")
