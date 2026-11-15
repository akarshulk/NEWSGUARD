
import os
import re
import ast
import json
import argparse
import joblib
import pandas as pd

from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

MODEL_DIR = os.path.join(ROOT, "models")

DATA_PATH = os.path.join(
    ROOT, "datasets", "raw", "labeled_dataset.xlsx"
)

BIAS_MODEL_PATH = os.path.join(
    MODEL_DIR, "bias_detection_model.pkl"
)

OPINION_MODEL_PATH = os.path.join(
    MODEL_DIR, "opinion_detection_model.pkl"
)

# ---------------------------------------------------------
# LABELS AND DISPLAY SETTINGS
# ---------------------------------------------------------

BIAS_LABELS = {
    0: "Unbiased",
    1: "Biased"
}

OPINION_LABELS = {
    0: "Entirely Factual",
    1: "Expresses Writer's Opinion",
    2: "Somewhat Factual but Opinionated"
}

# Provisional display guardrail.
# This is NOT a statistically calibrated threshold.
MIN_DISPLAY_CONFIDENCE = 60.0

# ---------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------

MODEL_DIR = os.path.join(ROOT, "models")

NESTED_CV_DIR = os.path.join(
    MODEL_DIR,
    "experimental",
    "nested_cv"
)

BIAS_MODEL_PRIMARY_PATH = os.path.join(
    NESTED_CV_DIR,
    "bias_detection_model_nestedcv.pkl"
)

OPINION_MODEL_PRIMARY_PATH = os.path.join(
    NESTED_CV_DIR,
    "opinion_detection_model_nestedcv.pkl"
)

# Original models are retained as fallback.
BIAS_MODEL_FALLBACK_PATH = os.path.join(
    MODEL_DIR,
    "bias_detection_model.pkl"
)

OPINION_MODEL_FALLBACK_PATH = os.path.join(
    MODEL_DIR,
    "opinion_detection_model.pkl"
)


def load_models():

    bias_path = (
        BIAS_MODEL_PRIMARY_PATH
        if os.path.exists(BIAS_MODEL_PRIMARY_PATH)
        else BIAS_MODEL_FALLBACK_PATH
    )

    opinion_path = (
        OPINION_MODEL_PRIMARY_PATH
        if os.path.exists(OPINION_MODEL_PRIMARY_PATH)
        else OPINION_MODEL_FALLBACK_PATH
    )

    if not os.path.exists(bias_path):
        raise FileNotFoundError(
            "Neither nested-CV nor original bias model was found."
        )

    if not os.path.exists(opinion_path):
        raise FileNotFoundError(
            "Neither nested-CV nor original opinion model was found."
        )

    bias_model = joblib.load(bias_path)
    opinion_model = joblib.load(opinion_path)

    print("Bias model loaded:", bias_path)
    print("Opinion model loaded:", opinion_path)

    return bias_model, opinion_model

# ---------------------------------------------------------
# EXTRACT DATASET TERMS
# ---------------------------------------------------------

def extract_terms(value):

    if pd.isna(value):
        return []

    if isinstance(value, (list, tuple, set)):
        items = list(value)

    elif isinstance(value, dict):
        items = list(value.keys())

    else:
        value = str(value).strip()

        try:
            parsed = ast.literal_eval(value)

            if isinstance(parsed, (list, tuple, set)):
                items = list(parsed)

            elif isinstance(parsed, dict):
                items = list(parsed.keys())

            else:
                items = re.split(r"[,;|\n]+", value)

        except (ValueError, SyntaxError):
            items = re.split(r"[,;|\n]+", value)

    terms = []

    for item in items:

        term = str(item).strip().strip("\"'[]{}()")
        term = re.sub(r"\s+", " ", term)

        if 3 <= len(term) <= 80:
            terms.append(term)

    return terms

def load_biased_terms():

    if not os.path.exists(DATA_PATH):
        print("Warning: MBIC dataset unavailable.")
        return []

    try:

        df = pd.read_excel(DATA_PATH)

        if "biased_words4" not in df.columns:
            return []

        term_counts = Counter()

        for value in df["biased_words4"].dropna():

            for term in extract_terms(value):
                term_counts[term.lower()] += 1

        selected = [
            term for term, count in term_counts.items()
            if count >= 2
        ]

        if not selected:
            selected = list(term_counts.keys())

        return sorted(selected, key=len, reverse=True)

    except Exception as error:

        print("Warning: Could not load MBIC terms:", error)
        return []

# ---------------------------------------------------------
# FILTER MEANINGFUL TERMS
# ---------------------------------------------------------

def is_meaningful_term(term):

    words = re.findall(
        r"\b[a-zA-Z]+\b",
        str(term).lower()
    )

    if not words:
        return False

    if all(word in ENGLISH_STOP_WORDS for word in words):
        return False

    if len(words) == 1:

        word = words[0]

        if word in ENGLISH_STOP_WORDS or len(word) < 4:
            return False

    return True

# ---------------------------------------------------------
# FIND BIASED TERMS
# ---------------------------------------------------------

def find_biased_terms(text, terms):

    found = []
    seen = set()

    text = str(text or "")

    for term in terms or []:

        if not is_meaningful_term(term):
            continue

        pattern = (
            r"(?<!\w)"
            + re.escape(term)
            + r"(?!\w)"
        )

        matches = re.findall(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if matches and term.lower() not in seen:

            found.append({
                "term": term,
                "count": len(matches)
            })

            seen.add(term.lower())

    return found

# ---------------------------------------------------------
# SENTENCE SEGMENTATION
# ---------------------------------------------------------

def split_into_sentences(text):
    """
    Split the article into sentence-sized inputs.

    MBIC training used individual sentences, so the models
    should receive inputs at approximately the same granularity.
    """

    text = str(text or "").strip()

    if not text:
        return []

    paragraphs = re.split(
        r"(?:\r?\n)+",
        text
    )

    sentences = []

    for paragraph in paragraphs:

        paragraph = re.sub(
            r"\s+",
            " ",
            paragraph
        ).strip()

        if not paragraph:
            continue

        # Protect common abbreviations before splitting.
        protected = paragraph

        protected = re.sub(
            r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs)\.",
            r"\1<ABBR>",
            protected,
            flags=re.IGNORECASE
        )

        protected = re.sub(
            r"\b(e\.g|i\.e)\.",
            r"\1<ABBR>",
            protected,
            flags=re.IGNORECASE
        )

        parts = re.split(
            r"(?<=[.!?])\s+(?=[A-Z0-9\"'“‘(])",
            protected
        )

        for part in parts:

            sentence = part.replace(
                "<ABBR>",
                "."
            ).strip()

            if sentence and re.search(
                r"[A-Za-z0-9]",
                sentence
            ):
                sentences.append(sentence)

    return sentences or [text]

# ---------------------------------------------------------
# MODEL PROBABILITIES
# ---------------------------------------------------------

def get_probability_map(model, text):
    """
    Return class probabilities where supported.

    No artificial confidence is generated for models
    that do not support predict_proba().
    """

    if not hasattr(model, "predict_proba"):
        return None

    probabilities = model.predict_proba([text])[0]

    classes = list(
        getattr(model, "classes_", [])
    )

    if len(classes) != len(probabilities):
        return None

    return {
        int(class_id): float(probability)
        for class_id, probability in zip(
            classes,
            probabilities
        )
    }

# ---------------------------------------------------------
# PREDICT ONE SENTENCE
# ---------------------------------------------------------

def predict_single_sentence(model, text, labels):

    raw_prediction = int(
        model.predict([text])[0]
    )

    probabilities = get_probability_map(
        model,
        text
    )

    if probabilities:

        predicted_class = max(
            probabilities,
            key=probabilities.get
        )

        confidence = (
            probabilities[predicted_class] * 100
        )

        raw_prediction = int(predicted_class)

        raw_label = labels.get(
            raw_prediction,
            "Unknown"
        )

        display_label = (
            raw_label
            if confidence >= MIN_DISPLAY_CONFIDENCE
            else "Inconclusive"
        )

    else:

        confidence = None

        raw_label = labels.get(
            raw_prediction,
            "Unknown"
        )

        display_label = "Needs review"

    return {
        "raw_prediction": raw_prediction,
        "raw_label": raw_label,
        "prediction": display_label,
        "confidence": (
            round(confidence, 2)
            if confidence is not None
            else None
        ),
        "probabilities": probabilities
    }

# ---------------------------------------------------------
# AGGREGATE SENTENCE PREDICTIONS
# ---------------------------------------------------------

def aggregate_sentence_predictions(
    model,
    sentences,
    labels
):

    sentence_results = []

    probability_totals = {
        int(class_id): 0.0
        for class_id in labels
    }

    probability_count = 0

    raw_counts = Counter()

    for index, sentence in enumerate(
        sentences,
        start=1
    ):

        result = predict_single_sentence(
            model,
            sentence,
            labels
        )

        raw_counts[result["raw_prediction"]] += 1

        probabilities = result["probabilities"]

        if probabilities:

            probability_count += 1

            for class_id in probability_totals:

                probability_totals[class_id] += (
                    probabilities.get(class_id, 0.0)
                )

        sentence_results.append({
            "Sentence No.": index,
            "Sentence": sentence,
            "Prediction": result["prediction"],
            "Raw Prediction": result["raw_label"],
            "Model Probability": (
                result["confidence"]
            )
        })

    # Average the sentence-level class probabilities.
    # This is a display aggregation, not a document-level model.
    if probability_count:

        averaged_probabilities = {
            class_id: total / probability_count
            for class_id, total in probability_totals.items()
        }

        overall_class = max(
            averaged_probabilities,
            key=averaged_probabilities.get
        )

        overall_confidence = (
            averaged_probabilities[overall_class] * 100
        )

        raw_label = labels.get(
            overall_class,
            "Unknown"
        )

        overall_prediction = (
            raw_label
            if overall_confidence >= MIN_DISPLAY_CONFIDENCE
            else "Inconclusive"
        )

        probability_distribution = {
            labels.get(class_id, str(class_id)): round(
                probability * 100,
                2
            )
            for class_id, probability
            in averaged_probabilities.items()
        }

    else:

        overall_class = raw_counts.most_common(1)[0][0]

        raw_label = labels.get(
            overall_class,
            "Unknown"
        )

        overall_confidence = None
        overall_prediction = "Needs review"
        probability_distribution = {}

    return {
        "prediction": overall_prediction,
        "raw_prediction": raw_label,
        "confidence": (
            round(overall_confidence, 2)
            if overall_confidence is not None
            else None
        ),
        "class_probabilities": probability_distribution,
        "sentences": sentence_results,
        "sentence_count": len(sentences),
        "probability_available": bool(probability_count)
    }

# ---------------------------------------------------------
# MAIN ANALYSIS FUNCTION
# ---------------------------------------------------------

def analyze_text(
    text,
    bias_model,
    opinion_model,
    biased_terms
):
    """
    Sentence-level bias and opinion analysis.

    The trained models are unchanged.
    """

    text = str(text or "").strip()

    if not text:
        raise ValueError(
            "Please provide news text."
        )

    sentences = split_into_sentences(text)

    # Bias analysis.
    bias_result = aggregate_sentence_predictions(
        bias_model,
        sentences,
        BIAS_LABELS
    )

    # Opinion analysis.
    opinion_result = aggregate_sentence_predictions(
        opinion_model,
        sentences,
        OPINION_LABELS
    )

    # Dataset-derived linguistic indicators.
    matched_terms = find_biased_terms(
        text,
        biased_terms
    )

    return {

        "bias_analysis": bias_result,

        "opinion_analysis": opinion_result,

        "linguistic_indicators": {
            "matched_terms": matched_terms,
            "total_matches": sum(
                item["count"]
                for item in matched_terms
            )
        },

        "interpretation": (
            "The MBIC models were trained on individual sentences. "
            "Results are therefore generated sentence by sentence. "
            "Overall probabilities are the mean of sentence-level "
            "probabilities. Low-confidence outputs are marked "
            "Inconclusive. Model probabilities are not guarantees "
            "of correctness. Dataset-derived indicators do not "
            "independently establish bias."
        ),

        "analysis_granularity": "sentence-level",

        "confidence_threshold": MIN_DISPLAY_CONFIDENCE
    }

# ---------------------------------------------------------
# COMMAND-LINE TESTING
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="NEWSGUARD AI Sentence-Level Bias Analysis"
    )

    parser.add_argument(
        "--text",
        type=str,
        help="News article or sentence"
    )

    args = parser.parse_args()

    print("=" * 55)
    print("NEWSGUARD AI - BIAS ANALYSIS")
    print("=" * 55)

    try:

        bias_model, opinion_model = load_models()

        biased_terms = load_biased_terms()

        print("\nModels loaded successfully.")
        print("Annotated terms loaded:", len(biased_terms))

        if args.text:

            news_text = args.text

        else:

            print(
                "\nEnter news text "
                "(press Enter twice to finish):"
            )

            lines = []

            while True:

                line = input()

                if not line:
                    break

                lines.append(line)

            news_text = " ".join(lines)

        result = analyze_text(
            news_text,
            bias_model,
            opinion_model,
            biased_terms
        )

        print("\nBIAS ANALYSIS RESULTS")

        print(
            json.dumps(
                result,
                indent=4,
                ensure_ascii=False
            )
        )

    except Exception as error:

        print("\nERROR:", error)

if __name__ == "__main__":
    main()