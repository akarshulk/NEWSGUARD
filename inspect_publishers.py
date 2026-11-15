
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "datasets" / "raw" / "WELFake_Dataset.csv"

df = pd.read_csv(DATA_PATH)

# Combine title and article text
df["content"] = (
    df["title"].fillna("").astype(str) + " " +
    df["text"].fillna("").astype(str)
).str.lower()

publishers = [
    "reuters",
    "breitbart",
    "cnn",
    "bbc",
    "new york times",
    "washington post",
    "fox news",
    "guardian",
    "associated press"
]

print("\n========== PUBLISHER DISTRIBUTION ==========")

for publisher in publishers:
    matching = df[df["content"].str.contains(
        publisher, regex=False, na=False
    )]

    if len(matching) == 0:
        continue

    fake = (matching["label"] == 0).sum()
    real = (matching["label"] == 1).sum()

    print(f"\nPublisher: {publisher}")
    print(f"Total articles: {len(matching)}")
    print(f"Fake-labelled: {fake}")
    print(f"Real-labelled: {real}")
    print(f"Fake percentage: {fake / len(matching) * 100:.2f}%")
    print(f"Real percentage: {real / len(matching) * 100:.2f}%")

print("\n========== COMPLETE ==========")
