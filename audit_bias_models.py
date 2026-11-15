
import os
import re
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    log_loss
)
from sklearn.dummy import DummyClassifier


ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    ROOT, "datasets", "raw", "labeled_dataset.xlsx"
)

MODEL_DIR = os.path.join(ROOT, "models")

TASKS = {
    "BIAS DETECTION": {
        "column": "Label_bias",
        "model": "bias_detection_model.pkl",
        "label_map": {
            "Non-biased": 0,
            "Biased": 1
        },
        "class_names": [
            "Unbiased",
            "Biased"
        ]
    },

    "OPINION DETECTION": {
        "column": "Label_opinion",
        "model": "opinion_detection_model.pkl",
        "label_map": {
            "Entirely factual": 0,
            "Expresses writer’s opinion": 1,
            "Somewhat factual but also opinionated": 2
        },
        "class_names": [
            "Entirely Factual",
            "Writer Opinion",
            "Factual and Opinionated"
        ]
    }
}


def normalize_sentence(text):
    """Normalize text for duplicate detection."""

    text = str(text or "").lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)

    return text.strip()


def audit_task(df, task_name, config):

    print("\n" + "=" * 70)
    print(task_name)
    print("=" * 70)

    data = df[
        ["sentence", config["column"]]
    ].copy()

    data["sentence"] = (
        data["sentence"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    data["label"] = (
        data[config["column"]]
        .astype(str)
        .str.strip()
        .map(config["label_map"])
    )

    data = data[
        (data["sentence"] != "")
        & data["label"].notna()
    ].copy()

    data["label"] = data["label"].astype(int)

    print("\nUsable records:", len(data))

    print("\nClass distribution:")
    print(
        data["label"]
        .value_counts()
        .sort_index()
        .rename(index=dict(enumerate(config["class_names"])))
    )

    # Check repeated sentences with conflicting labels.
    data["normalized"] = data["sentence"].apply(
        normalize_sentence
    )

    conflicting = (
        data.groupby("normalized")["label"]
        .nunique()
    )

    conflicting_count = int(
        (conflicting > 1).sum()
    )

    print(
        "\nUnique sentences having conflicting labels:",
        conflicting_count
    )

    # Recreate the original training split.
    X = data["sentence"]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Detect exact/normalized sentence overlap.
    train_texts = set(
        X_train.apply(normalize_sentence)
    )

    test_normalized = X_test.apply(normalize_sentence)

    overlap_count = int(
        test_normalized.isin(train_texts).sum()
    )

    print("\nTraining records:", len(X_train))
    print("Testing records:", len(X_test))

    print(
        "Test records also appearing in training:",
        overlap_count
    )

    # Load the existing trained model.
    model_path = os.path.join(
        MODEL_DIR,
        config["model"]
    )

    if not os.path.exists(model_path):
        print("Model not found:", model_path)
        return

    model = joblib.load(model_path)

    predictions = model.predict(X_test)

    class_ids = list(range(len(config["class_names"])))

    print("\n--- EXISTING MODEL ---")

    print(
        "Accuracy:",
        round(accuracy_score(y_test, predictions) * 100, 2),
        "%"
    )

    print(
        "Macro F1:",
        round(
            f1_score(
                y_test,
                predictions,
                average="macro",
                labels=class_ids,
                zero_division=0
            ),
            4
        )
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            labels=class_ids,
            target_names=config["class_names"],
            zero_division=0
        )
    )

    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=class_ids
        )
    )

    # Compare against a simple majority-class baseline.
    baseline = DummyClassifier(
        strategy="most_frequent"
    )

    baseline.fit(X_train, y_train)

    baseline_predictions = baseline.predict(X_test)

    print("\n--- MAJORITY-CLASS BASELINE ---")

    print(
        "Accuracy:",
        round(
            accuracy_score(y_test, baseline_predictions) * 100,
            2
        ),
        "%"
    )

    print(
        "Macro F1:",
        round(
            f1_score(
                y_test,
                baseline_predictions,
                average="macro",
                labels=class_ids,
                zero_division=0
            ),
            4
        )
    )

    # Inspect probability quality where supported.
    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(X_test)

        try:

            print(
                "\nLog Loss:",
                round(
                    log_loss(
                        y_test,
                        probabilities,
                        labels=model.classes_
                    ),
                    4
                )
            )

        except Exception as error:

            print("Log loss unavailable:", error)

    else:

        print(
            "\nProbability estimates are unavailable "
            "for this model."
        )


def main():

    print("=" * 70)
    print("NEWSGUARD AI - TRAINED MODEL AUDIT")
    print("=" * 70)

    if not os.path.exists(DATA_PATH):

        print("Dataset not found:", DATA_PATH)
        return

    df = pd.read_excel(DATA_PATH)

    print("\nDataset shape:", df.shape)

    if "sentence" not in df.columns:

        print("Required sentence column is missing.")
        return

    for task_name, config in TASKS.items():

        if config["column"] not in df.columns:

            print(
                f"\nSkipping {task_name}: "
                f"{config['column']} is missing."
            )

            continue

        try:

            audit_task(
                df,
                task_name,
                config
            )

        except Exception as error:

            print(
                f"\nAudit failed for {task_name}: {error}"
            )


if __name__ == "__main__":
    main()
