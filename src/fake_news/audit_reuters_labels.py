
import re
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "datasets" / "raw" / "WELFake_Dataset.csv"

df = pd.read_csv(DATA_PATH)

df = df.dropna(subset=["title", "text", "label"]).copy()

# Find Reuters mentions in the original article text.
mask = (
    df["title"].astype(str).str.contains(
        r"\bReuters\b", case=False, regex=True
    )
    |
    df["text"].astype(str).str.contains(
        r"\bReuters\b", case=False, regex=True
    )
)

reuters_df = df[mask].copy()

# Separate the two dataset labels.
fake = reuters_df[reuters_df["label"] == 0]
real = reuters_df[reuters_df["label"] == 1]

# Reproducible random samples.
fake_sample = fake.sample(
    n=min(10, len(fake)),
    random_state=42
)

real_sample = real.sample(
    n=min(10, len(real)),
    random_state=42
)

audit = pd.concat(
    [fake_sample, real_sample]
)

audit = audit[["title", "text", "label"]].copy()

audit["dataset_label"] = audit["label"].map({
    0: "FAKE",
    1: "REAL"
})

audit.drop(columns=["label"], inplace=True)

output_path = ROOT / "datasets" / "processed" / "reuters_audit_sample.csv"

output_path.parent.mkdir(parents=True, exist_ok=True)

audit.to_csv(output_path, index=False)

print("Reuters audit sample created.")
print("Fake-labelled records:", len(fake_sample))
print("Real-labelled records:", len(real_sample))
print("Saved to:", output_path)
