import pandas as pd
import xgboost as xgb
from feature_extractor import URLFeatureExtractor

# -----------------------------
# Same FEATURE_ORDER as training
# -----------------------------
FEATURE_ORDER = [
    'url_len',
    'dot_count',
    'hyphen_count',
    'has_ip',
    'suspicious_words',
    'subdomain_count',
    'tld_length',
    'url_entropy',
    'has_a',
    'has_mx',
    'has_ns',
    'ip_count'
]

# -----------------------------
# Load model & training stats
# -----------------------------
print("📂 Loading model...")
model = xgb.XGBClassifier()
model.load_model("url_xgb_model.json")

# Load your training feature file to compute averages (optional but recommended)
print("📊 Loading training feature data...")
train_df = pd.read_csv("url_dataset_balanced.csv").sample(n=10000, random_state=42).reset_index(drop=True)

# Extract features for training set (averages)
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def safe_extract(url):
    extractor = URLFeatureExtractor(url)
    try:
        has_a, has_mx, has_ns, ip_count = extractor.get_dns_info()
        return {
            'url_len': extractor.url_length(),
            'dot_count': extractor.count_dots(),
            'hyphen_count': extractor.count_hyphens(),
            'has_ip': extractor.has_ip(),
            'suspicious_words': extractor.count_suspicious_words(),
            'subdomain_count': extractor.subdomain_count(),
            'tld_length': extractor.tld_length(),
            'url_entropy': extractor.url_entropy(),
            'has_a': has_a,
            'has_mx': has_mx,
            'has_ns': has_ns,
            'ip_count': ip_count
        }
    except:
        return None

print("⚙️ Extracting training features for averages...")
with ThreadPoolExecutor(max_workers=4) as ex:
    train_features = list(tqdm(ex.map(safe_extract, train_df['url']), total=len(train_df)))

train_df = train_df.assign(features=train_features)
train_df = train_df[train_df['features'].notnull()]

# Build features DataFrame
features_expanded = pd.DataFrame(list(train_df['features']))
labels = train_df['label'].reset_index(drop=True)

avg_benign = features_expanded[labels == 0].mean()
avg_malicious = features_expanded[labels == 1].mean()

# -----------------------------
# Test URLs
# -----------------------------
test_urls = [
    "http://secure-login-update.com",
    "https://www.google.com",
    "http://free-money-offer.com",
    "https://bankofamerica.com",
    "https://www.prydwen.gg/",
    "https://shopee.ph/",
    "https://funky-article-722279.framer.app/",
    "https://special-offers-signup.att.com/#/237/HRApplicationOffer",
    "https://www.geeksforgeeks.org/",
    "https://docs.google.com/document/d/1R-tJAzhftiZeIcn-mMwCbIOOxkjupdfDFvP_dLG1u1g/edit",
    "https://housitba5.firebaseapp.com/",
    "http://slatteryauctions.com.au"
]

# -----------------------------
# Diagnose each URL
# -----------------------------
print("\n🔎 Diagnostic predictions:")
for url in test_urls:
    extractor = URLFeatureExtractor(url)
    try:
        has_a, has_mx, has_ns, ip_count = extractor.get_dns_info()
        feat_dict = {
            'url_len': extractor.url_length(),
            'dot_count': extractor.count_dots(),
            'hyphen_count': extractor.count_hyphens(),
            'has_ip': extractor.has_ip(),
            'suspicious_words': extractor.count_suspicious_words(),
            'subdomain_count': extractor.subdomain_count(),
            'tld_length': extractor.tld_length(),
            'url_entropy': extractor.url_entropy(),
            'has_a': has_a,
            'has_mx': has_mx,
            'has_ns': has_ns,
            'ip_count': ip_count
        }

        # Compare with training averages
        print(f"\nURL: {url}")
        for feat in FEATURE_ORDER:
            print(f"  {feat:<17} = {feat_dict[feat]:<8} "
                  f"(avg benign: {avg_benign[feat]:.2f}, avg malicious: {avg_malicious[feat]:.2f})")

        # Prediction
        df = pd.DataFrame([[feat_dict[f] for f in FEATURE_ORDER]], columns=FEATURE_ORDER)
        proba = model.predict_proba(df)[0][1]
        label = "🔴 Malicious" if proba >= 0.5 else "🟢 Benign"
        print(f"→ Prediction: {label} ({proba * 100:.2f}% confidence)")

    except Exception as e:
        print(f"{url} → ❌ Feature extraction failed: {e}")
