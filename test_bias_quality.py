
import sys
import pandas as pd

sys.path.insert(0, "src")

from bias_detection.predict_bias import (
    load_models,
    load_biased_terms,
    analyze_text
)

bias_model, opinion_model = load_models()
biased_terms = load_biased_terms()

test_sentences = [
    "Benjamin Netanyahu is the prime minister of Israel.",
    "The government announced three new policies yesterday.",
    "The government completely failed its citizens.",
    "The government's disastrous policies have devastated ordinary families.",
    "According to official records, the policy was introduced in 2024."
]

results = []

for sentence in test_sentences:

    result = analyze_text(
        sentence,
        bias_model,
        opinion_model,
        biased_terms
    )

    bias = result["bias_analysis"]
    opinion = result["opinion_analysis"]

    results.append({
        "Sentence": sentence,
        "Bias Prediction": bias["prediction"],
        "Bias Probability": bias["confidence"],
        "Opinion Prediction": opinion["prediction"],
        "Opinion Probability": opinion["confidence"],
        "Matched Indicators": result["linguistic_indicators"]["total_matches"]
    })

df = pd.DataFrame(results)

print(df.to_string(index=False))
