
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "datasets" / "raw" / "WELFake_Dataset.csv"

df = pd.read_csv(DATA_PATH)

df["title"] = df["title"].fillna("").astype(str)
df["text"] = df["text"].fillna("").astype(str)

for column in ["title", "text"]:
    matches = df[df[column].str.contains(
        "reuters", case=False, regex=False, na=False
    )]

    print(f"\n========== REUTERS IN {column.upper()} ==========")
    print("Total:", len(matches))
    print("Fake-labelled:", (matches["label"] == 0).sum())
    print("Real-labelled:", (matches["label"] == 1).sum())

    print("\nSample articles:")
    for _, row in matches.head(5).iterrows():
        print("\nLabel:", row["label"])
        print("Title:", row["title"][:250])
        print("Text:", row["text"][:350])

print("\n========== COMPLETE ==========")
