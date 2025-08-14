import pandas as pd

# -----------------------------
# Load PhishTank malicious URLs
# -----------------------------
df_phish = pd.read_csv("raw_datasets/malicious-urls.csv")
df_phish = df_phish[["url"]].drop_duplicates()
df_phish["label"] = 1  # malicious
df_phish["popularity_rank"] = 1_000_000  # Assume worst rank for malicious

# -----------------------------
# Load Tranco benign domains
# -----------------------------
df_tranco = pd.read_csv("raw_datasets/benign-urls.csv", header=None, names=["rank", "domain"])
df_tranco["url"] = "http://" + df_tranco["domain"]
df_tranco = df_tranco[["url", "rank"]].drop_duplicates()
df_tranco["label"] = 0  # benign
df_tranco.rename(columns={"rank": "popularity_rank"}, inplace=True)

# -----------------------------
# Combine datasets (no sampling yet)
# -----------------------------
df_combined = pd.concat([df_phish, df_tranco], ignore_index=True)
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

# -----------------------------
# Save for feature extraction
# -----------------------------
df_combined.to_csv("url_dataset_with_rank.csv", index=False)
print(f"✅ Combined and saved: {len(df_combined)} entries (with popularity_rank feature)")
