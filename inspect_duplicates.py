
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "datasets" / "raw" / "WELFake_Dataset.csv"

df = pd.read_csv(DATA_PATH)

df["title"] = df["title"].fillna("").astype(str).str.strip().str.lower()
df["text"] = df["text"].fillna("").astype(str).str.strip().str.lower()

# Exact duplicate article bodies
body_counts = df.groupby("text")["label"].agg(
    count="size",
    unique_labels="nunique"
)

duplicates = body_counts[body_counts["count"] > 1]
conflicts = body_counts[body_counts["unique_labels"] > 1]

print("\n========== DUPLICATE ANALYSIS ==========")
print("Duplicate article bodies:", len(duplicates))
print("Duplicate bodies with conflicting labels:", len(conflicts))

# Exact duplicate title + body combinations
df["combined"] = df["title"] + " " + df["text"]

combined_counts = df.groupby("combined")["label"].agg(
    count="size",
    unique_labels="nunique"
)

print("\nDuplicate title + body combinations:",
      (combined_counts["count"] > 1).sum())

print("Conflicting title + body labels:",
      (combined_counts["unique_labels"] > 1).sum())

print("\n========== COMPLETE ==========")
