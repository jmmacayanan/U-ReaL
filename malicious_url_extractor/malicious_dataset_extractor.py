import pandas as pd
import requests
from io import StringIO

# Download CSV
url = "http://data.phishtank.com/data/online-valid.csv"
response = requests.get(url)
if response.status_code != 200:
    raise Exception("Failed to download data from PhishTank")

# Read CSV into DataFrame
csv_data = StringIO(response.text)
df_phish = pd.read_csv(
    csv_data,
    engine='python',
    quotechar='"',
    skipinitialspace=True,
    on_bad_lines='skip'
)


print(df_phish.columns.tolist())

# Optional: Drop duplicates or select subset
df_phish = df_phish.drop_duplicates(subset="url")
df_phish = df_phish[["url"]].copy()
df_phish["label"] = 1  # 1 = malicious

print("✅ Loaded PhishTank URLs:", len(df_phish))

df_phish.to_csv("phishtank_urls.csv", index=False)
