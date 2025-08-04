if __name__ == '__main__':
    import pandas as pd
    # import math
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    import concurrent.futures
    from tqdm import tqdm
    from feature_extractor import URLFeatureExtractor

    def safe_extract_features(index_url):
        i, url = index_url
        extractor = URLFeatureExtractor(url)
        features = extractor.extract_features()
        if features is None:
            return None
        return i, features

    df = pd.read_csv("url_dataset_balanced.csv").sample(n=12000, random_state=42).reset_index(drop=True)
    print(df['label'].value_counts())

    print("⚙️ Extracting features...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(tqdm(executor.map(safe_extract_features, enumerate(df['url'])), total=len(df)))

    valid_results = [res for res in results if res is not None]
    if not valid_results:
        raise Exception("❌ No valid URLs were processed. Check your dataset or internet connection.")

    indices, feature_rows = zip(*valid_results)
    features_df = pd.DataFrame(feature_rows)
    labels = df.loc[list(indices), 'label'].reset_index(drop=True)

    X_temp, X_test, y_temp, y_test = train_test_split(features_df, labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        early_stopping_rounds=10,
        colsample_bytree=0.8,
        eval_metric='logloss',
        verbosity=1
    )

    print("🚀 Training model...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )

    y_pred = model.predict(X_test)
    print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    model.save_model("url_xgb_model.json")
    print("✅ Model saved as url_xgb_model.json")
