
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "datasets" / "raw" / "WELFake_Dataset.csv"

df = pd.read_csv(DATA_PATH)

# Normalize text consistently
df["title_clean"] = (
    df["title"].fillna("").astype(str).str.lower().str.strip()
)

df["text_clean"] = (
    df["text"].fillna("").astype(str).str.lower().str.strip()
)

# Reproduce the original 80/20 stratified split
train_idx, test_idx = train_test_split(
    df.index,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

train = df.loc[train_idx]
test = df.loc[test_idx]

def check_overlap(column):
    train_values = set(train[column])
    test_values = test[column]

    overlap = test_values.isin(train_values)

    print(f"\n{column}")
    print("Unique training values:", len(train_values))
    print("Test rows:", len(test_values))
    print("Test rows also present in training:", overlap.sum())
    print("Percentage of test rows affected:",
          round(overlap.mean() * 100, 2), "%")

print("\n========== TRAIN-TEST LEAKAGE ==========")

check_overlap("text_clean")

df["combined_clean"] = (
    df["title_clean"] + " " + df["text_clean"]
)

# Rebuild split using the same row indices
train = df.loc[train_idx]
test = df.loc[test_idx]

check_overlap("combined_clean")

print("\n========== COMPLETE ==========")
