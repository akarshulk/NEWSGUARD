
import pandas as pd
from pathlib import Path

# Dataset directory
DATA_DIR = Path("datasets/raw")

# Locate datasets
csv_files = list(DATA_DIR.glob("*.csv"))
excel_files = list(DATA_DIR.glob("*.xlsx"))

print("=" * 55)
print("       NEWSGUARD AI - DATASET INSPECTION")
print("=" * 55)

# 1. Inspect Fake News Dataset
print("\n[1] FAKE NEWS DATASET")

if csv_files:
    fake_df = pd.read_csv(csv_files[0], encoding="utf-8", on_bad_lines="skip")

    print("File:", csv_files[0].name)
    print("Shape:", fake_df.shape)
    print("Columns:", fake_df.columns.tolist())
    print("\nMissing Values:\n", fake_df.isnull().sum())
    print("\nDuplicate Rows:", fake_df.duplicated().sum())

    if "label" in fake_df.columns:
        print("\nLabel Distribution:\n", fake_df["label"].value_counts())
else:
    print("No CSV file found!")

# 2. Inspect Bias Dataset
print("\n[2] BIAS DETECTION DATASET")

if excel_files:
    bias_df = pd.read_excel(excel_files[0])

    print("File:", excel_files[0].name)
    print("Shape:", bias_df.shape)
    print("Columns:", bias_df.columns.tolist())
    print("\nMissing Values:\n", bias_df.isnull().sum())
    print("\nFirst 5 Rows:\n", bias_df.head())
else:
    print("No Excel file found!")

print("\n" + "=" * 55)
print("       DATASET INSPECTION COMPLETED")
print("=" * 55)
