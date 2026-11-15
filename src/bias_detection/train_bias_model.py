import os
import json
import joblib
import pandas as pd

from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    log_loss
)


# =========================================================
# PATHS
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "datasets" / "raw" / "labeled_dataset.xlsx"
MODEL_DIR = ROOT / "models"
EXPERIMENT_DIR = MODEL_DIR / "experimental"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# LABEL MAPPINGS
# =========================================================

BIAS_MAP = {
    "Non-biased": 0,
    "Biased": 1
}

OPINION_MAP = {
    "Entirely factual": 0,
    "Expresses writer’s opinion": 1,
    "Somewhat factual but also opinionated": 2
}

TASKS = [
    (
        "Label_bias",
        "bias_detection_model.pkl",
        BIAS_MAP,
        ["Unbiased", "Biased"]
    ),
    (
        "Label_opinion",
        "opinion_detection_model.pkl",
        OPINION_MAP,
        [
            "Entirely Factual",
            "Writer Opinion",
            "Factual and Opinionated"
        ]
    )
]


# =========================================================
# IMPROVED FEATURE EXTRACTION
# =========================================================

def build_pipeline():

    features = FeatureUnion([
        (
            "word",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                stop_words=None,
                sublinear_tf=True,
                ngram_range=(1, 2),
                min_df=1,
                max_features=25000
            )
        ),

        (
            "char",
            TfidfVectorizer(
                analyzer="char_wb",
                lowercase=True,
                strip_accents="unicode",
                sublinear_tf=True,
                ngram_range=(3, 5),
                min_df=2,
                max_features=30000
            )
        )
    ])

    pipeline = Pipeline([
        ("features", features),

        (
            "classifier",
            LogisticRegression(
                max_iter=3000,
                solver="lbfgs"
            )
        )
    ])

    return pipeline


# =========================================================
# EVALUATION
# =========================================================

def evaluate_model(name, model, X_test, y_test, class_names):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, predictions)

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro"
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted"
    )

    loss = log_loss(
        y_test,
        probabilities,
        labels=list(range(len(class_names)))
    )

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(f"Accuracy:       {accuracy * 100:.2f}%")
    print(f"Macro F1:       {macro_f1:.4f}")
    print(f"Weighted F1:    {weighted_f1:.4f}")
    print(f"Log Loss:       {loss:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            labels=list(range(len(class_names))),
            target_names=class_names,
            zero_division=0
        )
    )

    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=list(range(len(class_names)))
        )
    )

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "log_loss": float(loss)
    }


# =========================================================
# TRAIN AND COMPARE
# =========================================================

def train_task(
    df,
    target_column,
    model_name,
    label_map,
    class_names
):

    print("\n\n" + "#" * 65)
    print("TASK:", target_column)
    print("#" * 65)

    # -----------------------------------------------------
    # DATA PREPARATION
    # -----------------------------------------------------

    data = df[["sentence", target_column]].copy()

    data["sentence"] = (
        data["sentence"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    data["label"] = (
        data[target_column]
        .astype(str)
        .str.strip()
        .map(label_map)
    )

    data = data[
        (data["sentence"] != "") &
        (data["label"].notna())
    ].copy()

    data["label"] = data["label"].astype(int)

    X = data["sentence"]
    y = data["label"]

    print("\nUsable records:", len(data))

    print("\nClass distribution:")
    print(y.value_counts().sort_index())

    if y.nunique() != len(class_names):
        print("ERROR: One or more expected classes are missing.")
        return

    if y.value_counts().min() < 5:
        print("ERROR: Insufficient examples for 5-fold CV.")
        return

    # -----------------------------------------------------
    # SAME HOLDOUT SPLIT AS THE PREVIOUS EXPERIMENT
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining records:", len(X_train))
    print("Testing records:", len(X_test))

    # -----------------------------------------------------
    # LOAD EXISTING MODEL
    # -----------------------------------------------------

    old_model_path = MODEL_DIR / model_name

    if not old_model_path.exists():
        print("Existing model not found:", old_model_path)
        return

    old_model = joblib.load(old_model_path)

    baseline_metrics = evaluate_model(
        "EXISTING MODEL",
        old_model,
        X_test,
        y_test,
        class_names
    )

    # -----------------------------------------------------
    # BUILD IMPROVED MODEL
    # -----------------------------------------------------

    pipeline = build_pipeline()

    param_grid = {
        "features__word__ngram_range": [
            (1, 2),
            (1, 3)
        ],

        "classifier__C": [
            0.3,
            1,
            3
        ],

        "classifier__class_weight": [
            None,
            "balanced"
        ]
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True
    )

    print("\nTraining improved TF-IDF model...")

    search.fit(X_train, y_train)

    candidate_model = search.best_estimator_

    print("\nBest CV Macro F1:", round(search.best_score_, 4))

    print("\nBest Parameters:")

    print(search.best_params_)

    # -----------------------------------------------------
    # EVALUATE CANDIDATE
    # -----------------------------------------------------

    candidate_metrics = evaluate_model(
        "IMPROVED CANDIDATE MODEL",
        candidate_model,
        X_test,
        y_test,
        class_names
    )

    # -----------------------------------------------------
    # COMPARE
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    comparison = pd.DataFrame([
        {
            "Model": "Existing",
            **baseline_metrics
        },
        {
            "Model": "Improved Candidate",
            **candidate_metrics
        }
    ])

    print(comparison.to_string(index=False))

    f1_difference = (
        candidate_metrics["macro_f1"] -
        baseline_metrics["macro_f1"]
    )

    print("\nMacro F1 improvement:", round(f1_difference, 4))

    # -----------------------------------------------------
    # SAVE CANDIDATE SEPARATELY
    # -----------------------------------------------------

    candidate_name = model_name.replace(
        ".pkl",
        "_candidate.pkl"
    )

    candidate_path = EXPERIMENT_DIR / candidate_name

    joblib.dump(candidate_model, candidate_path)

    report = {
        "task": target_column,
        "existing_model": baseline_metrics,
        "candidate_model": candidate_metrics,
        "cv_macro_f1": float(search.best_score_),
        "best_parameters": search.best_params_,
        "macro_f1_difference": float(f1_difference),
        "candidate_path": str(candidate_path)
    }

    report_path = EXPERIMENT_DIR / model_name.replace(
        ".pkl",
        "_comparison.json"
    )

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\nCandidate saved:", candidate_path)
    print("Comparison report saved:", report_path)

    if f1_difference > 0:
        print("\nCandidate has a higher holdout Macro F1.")
    elif f1_difference < 0:
        print("\nExisting model has a higher holdout Macro F1.")
    else:
        print("\nBoth models have equal holdout Macro F1.")

    print("\nOriginal model has NOT been overwritten.")


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 65)
    print("NEWSGUARD AI - IMPROVED BIAS & OPINION MODEL EXPERIMENT")
    print("=" * 65)

    if not DATA_PATH.exists():
        print("\nDataset not found:", DATA_PATH)
        return

    df = pd.read_excel(DATA_PATH)

    print("\nDataset shape:", df.shape)

    required_columns = {
        "sentence",
        "Label_bias",
        "Label_opinion"
    }

    missing = required_columns - set(df.columns)

    if missing:
        print("\nMissing dataset columns:", missing)
        return

    for task in TASKS:
        train_task(df, *task)

    print("\n" + "=" * 65)
    print("EXPERIMENT COMPLETED")
    print("Original model files remain unchanged.")
    print("=" * 65)


if __name__ == "__main__":
    main()