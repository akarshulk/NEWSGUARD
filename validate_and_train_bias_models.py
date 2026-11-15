
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline, FeatureUnion


# =========================================================
# CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parent

DATA_PATH = ROOT / "datasets" / "raw" / "labeled_dataset.xlsx"

OUTPUT_DIR = ROOT / "models" / "experimental" / "nested_cv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTER_FOLDS = 5
INNER_FOLDS = 4
RANDOM_STATE = 42


TASKS = [
    (
        "Label_bias",
        "bias_detection_model_nestedcv.pkl",
        {"Non-biased": 0, "Biased": 1},
        ["Unbiased", "Biased"]
    ),
    (
        "Label_opinion",
        "opinion_detection_model_nestedcv.pkl",
        {
            "Entirely factual": 0,
            "Expresses writer’s opinion": 1,
            "Somewhat factual but also opinionated": 2
        },
        [
            "Entirely Factual",
            "Writer Opinion",
            "Factual and Opinionated"
        ]
    )
]


# =========================================================
# GROUP NORMALIZATION
# =========================================================

def normalize_group(text):

    text = str(text).lower().strip()

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " URL ",
        text
    )

    text = re.sub(r"[^\w]+", " ", text)

    return re.sub(r"\s+", " ", text).strip()


# =========================================================
# MODEL PIPELINE
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

    return Pipeline([
        ("features", features),
        (
            "classifier",
            LogisticRegression(
                max_iter=3000,
                solver="lbfgs",
                random_state=RANDOM_STATE
            )
        )
    ])


PARAM_GRID = {
    "features__word__ngram_range": [
        (1, 2),
        (1, 3)
    ],
    "classifier__C": [
        0.3,
        1.0
    ],
    "classifier__class_weight": [
        None,
        "balanced"
    ]
}


# =========================================================
# NESTED CROSS-VALIDATION
# =========================================================

def evaluate_task(X, y, groups, class_names, task_name):

    if y.nunique() != len(class_names):
        raise ValueError(f"Missing classes in {task_name}")

    if y.value_counts().min() < OUTER_FOLDS:
        raise ValueError(f"Insufficient class examples in {task_name}")

    outer_cv = StratifiedGroupKFold(
        n_splits=OUTER_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    n = len(X)
    k = len(class_names)

    oof_predictions = np.full(n, -1, dtype=int)
    oof_probabilities = np.full((n, k), np.nan)
    baseline_predictions = np.full(n, -1, dtype=int)

    fold_results = []

    print(
        f"\nRunning {OUTER_FOLDS}-fold nested "
        f"cross-validation..."
    )

    print(
        "Unique normalized groups:",
        len(np.unique(groups))
    )

    for fold, (train_idx, test_idx) in enumerate(
        outer_cv.split(X, y, groups), start=1
    ):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        train_groups = groups[train_idx]

        if set(train_groups) & set(groups[test_idx]):
            raise RuntimeError("Duplicate sentence leakage detected.")

        inner_cv = StratifiedGroupKFold(
            n_splits=INNER_FOLDS,
            shuffle=True,
            random_state=RANDOM_STATE + fold
        )

        search = GridSearchCV(
            estimator=build_pipeline(),
            param_grid=PARAM_GRID,
            scoring="f1_macro",
            cv=inner_cv,
            n_jobs=-1,
            refit=True,
            error_score="raise"
        )

        search.fit(
            X_train,
            y_train,
            groups=train_groups
        )

        best_model = search.best_estimator_

        predictions = best_model.predict(X_test)

        probabilities = best_model.predict_proba(X_test)

        oof_predictions[test_idx] = predictions

        oof_probabilities[test_idx] = probabilities

        # Majority-class baseline
        dummy = DummyClassifier(strategy="most_frequent")

        dummy.fit(
            X_train.to_frame(),
            y_train
        )

        baseline_predictions[test_idx] = dummy.predict(
            X_test.to_frame()
        )

        fold_accuracy = accuracy_score(
            y_test,
            predictions
        )

        fold_f1 = f1_score(
            y_test,
            predictions,
            average="macro"
        )

        fold_results.append({
            "fold": fold,
            "accuracy": float(fold_accuracy),
            "macro_f1": float(fold_f1),
            "inner_cv_macro_f1": float(search.best_score_),
            "best_parameters": search.best_params_
        })

        print(
            f"Fold {fold}: "
            f"Accuracy={fold_accuracy:.4f}, "
            f"Macro F1={fold_f1:.4f}"
        )

    if (oof_predictions < 0).any():
        raise RuntimeError("Missing out-of-fold predictions.")

    if np.isnan(oof_probabilities).any():
        raise RuntimeError("Missing out-of-fold probabilities.")

    labels = list(range(k))

    accuracy = accuracy_score(y, oof_predictions)

    macro_f1 = f1_score(
        y, oof_predictions, average="macro"
    )

    weighted_f1 = f1_score(
        y, oof_predictions, average="weighted"
    )

    loss = log_loss(
        y,
        oof_probabilities,
        labels=labels
    )

    baseline_accuracy = accuracy_score(
        y, baseline_predictions
    )

    baseline_f1 = f1_score(
        y, baseline_predictions, average="macro"
    )

    matrix = confusion_matrix(
        y, oof_predictions, labels=labels
    )

    fold_f1 = [row["macro_f1"] for row in fold_results]

    print("\n========== NESTED CV RESULTS ==========")

    print(f"Accuracy:    {accuracy * 100:.2f}%")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Log Loss:    {loss:.4f}")

    print(
        f"Fold Macro F1: {np.mean(fold_f1):.4f} "
        f"+/- {np.std(fold_f1, ddof=1):.4f}"
    )

    print("\nMajority Baseline:")

    print(f"Accuracy: {baseline_accuracy * 100:.2f}%")
    print(f"Macro F1: {baseline_f1:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y,
            oof_predictions,
            labels=labels,
            target_names=class_names,
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(matrix)

    report = {
        "task": task_name,
        "rows": len(X),
        "unique_groups": len(np.unique(groups)),
        "pooled_oof": {
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "log_loss": float(loss)
        },
        "fold_macro_f1_mean": float(np.mean(fold_f1)),
        "fold_macro_f1_std": float(np.std(fold_f1, ddof=1)),
        "majority_baseline": {
            "accuracy": float(baseline_accuracy),
            "macro_f1": float(baseline_f1)
        },
        "classification_report": classification_report(
            y,
            oof_predictions,
            labels=labels,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        ),
        "confusion_matrix": matrix.tolist(),
        "fold_results": fold_results
    }

    return (
        report,
        oof_predictions,
        oof_probabilities,
        baseline_predictions
    )


# =========================================================
# TRAIN FINAL CANDIDATE ON ALL LABELLED DATA
# =========================================================

def train_final_candidate(X, y, groups, model_filename):

    cv = StratifiedGroupKFold(
        n_splits=INNER_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    search = GridSearchCV(
        estimator=build_pipeline(),
        param_grid=PARAM_GRID,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        refit=True,
        error_score="raise"
    )

    print("\nTraining final candidate on all labelled data...")

    search.fit(
        X,
        y,
        groups=groups
    )

    model_path = OUTPUT_DIR / model_filename

    joblib.dump(
        search.best_estimator_,
        model_path
    )

    print("Candidate saved:", model_path)
    print("Selected parameters:", search.best_params_)

    return {
        "candidate_path": str(model_path),
        "full_data_cv_score_for_selection_only": float(
            search.best_score_
        ),
        "best_parameters": search.best_params_
    }


# =========================================================
# MAIN
# =========================================================

def main():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_excel(DATA_PATH)

    required = {
        "sentence",
        "Label_bias",
        "Label_opinion"
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    print("NEWSGUARD AI - NESTED MODEL VALIDATION")
    print("Dataset shape:", df.shape)

    for target, model_filename, label_map, class_names in TASKS:

        print("\n" + "=" * 65)
        print("TASK:", target)
        print("=" * 65)

        data = df[["sentence", target]].copy()

        data["sentence"] = (
            data["sentence"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        data["label"] = (
            data[target]
            .astype(str)
            .str.strip()
            .map(label_map)
        )

        data = data[
            (data["sentence"] != "") &
            data["label"].notna()
        ].reset_index(drop=True)

        data["label"] = data["label"].astype(int)

        X = data["sentence"]
        y = data["label"]

        groups = X.map(normalize_group).to_numpy()

        # Detect conflicting labels for normalized duplicate sentences.
        group_counts = (
            data.assign(group=groups)
            .groupby("group")["label"]
            .nunique()
        )

        if (group_counts > 1).any():
            raise ValueError(
                "Conflicting labels found for identical normalized "
                f"sentences in {target}."
            )

        print("Usable records:", len(data))
        print("Class distribution:")
        print(y.value_counts().sort_index())

        (
            report,
            predictions,
            probabilities,
            baseline_predictions
        ) = evaluate_task(
            X, y, groups, class_names, target
        )

        candidate_info = train_final_candidate(
            X,
            y,
            groups,
            model_filename
        )

        report["final_candidate"] = candidate_info

        # Export out-of-fold predictions for error analysis.
        prediction_data = pd.DataFrame({
            "sentence": X,
            "actual_label": y,
            "predicted_label": predictions,
            "majority_baseline": baseline_predictions
        })

        for i, label in enumerate(class_names):

            safe_label = re.sub(
                r"[^a-z0-9]+",
                "_",
                label.lower()
            ).strip("_")

            prediction_data[
                f"probability_{safe_label}"
            ] = probabilities[:, i]

        prediction_path = (
            OUTPUT_DIR / f"{target.lower()}_oof_predictions.csv"
        )

        prediction_data.to_csv(
            prediction_path,
            index=False,
            encoding="utf-8-sig"
        )

        report_path = (
            OUTPUT_DIR / f"{target.lower()}_nested_cv_report.json"
        )

        report_path.write_text(
            json.dumps(
                report,
                indent=4,
                ensure_ascii=False,
                default=str
            ),
            encoding="utf-8"
        )

        print("\nReport:", report_path)
        print("OOF predictions:", prediction_path)

    print("\nEXPERIMENT COMPLETED")
    print("Existing production models were not modified.")


if __name__ == "__main__":
    main()
