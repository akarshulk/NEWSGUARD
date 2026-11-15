
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "datasets" / "raw" / "WELFake_Dataset.csv"

df = pd.read_csv(DATA_PATH)

df["text_clean"] = (
    df["text"].fillna("")
    .astype(str).str.lower().str.strip()
)

conflicting = (
    df.groupby("text_clean")["label"]
    .nunique()
)

conflicting_bodies = conflicting[conflicting > 1].index

print("Conflicting article bodies:", len(conflicting_bodies))

for body in conflicting_bodies:
    rows = df[df["text_clean"] == body]

    print("\n========== CONFLICT ==========")
    print("Labels:", rows["label"].tolist())

    for _, row in rows.iterrows():
        print("Title:", row["title"])
        print("Label:", row["label"])
        print("Article:", str(row["text"])[:500])
