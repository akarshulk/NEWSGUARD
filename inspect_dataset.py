import pandas as pd

DATA_PATH = "datasets/raw/WELFake_Dataset.csv"

df = pd.read_csv(DATA_PATH)

print("\n========== DATASET OVERVIEW ==========")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

print("\n========== LABEL DISTRIBUTION ==========")
print(df["label"].value_counts(dropna=False))

print("\n========== MISSING VALUES ==========")
print(df[["title", "text", "label"]].isnull().sum())

print("\n========== SAMPLE FAKE NEWS (LABEL 0) ==========")

fake = df[df["label"] == 0]

for _, row in fake.head(5).iterrows():
    print("\nTitle:", row["title"])
    print("Article:", str(row["text"])[:400])
    print("-" * 60)

print("\n========== SAMPLE REAL NEWS (LABEL 1) ==========")

real = df[df["label"] == 1]

for _, row in real.head(5).iterrows():
    print("\nTitle:", row["title"])
    print("Article:", str(row["text"])[:400])
    print("-" * 60)

print("\n========== SHORT ARTICLE ANALYSIS ==========")

df["text_length"] = df["text"].fillna("").str.split().str.len()

print(
    df.groupby("label")["text_length"]
    .describe()
    .round(2)
)

print("\n========== DIAGNOSTIC COMPLETE ==========")