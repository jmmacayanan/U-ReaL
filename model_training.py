if __name__ == '__main__':
    import pandas as pd
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    import concurrent.futures
    from tqdm import tqdm
    from feature_extractor import URLFeatureExtractor

    # -------- Feature extraction helper --------
    def safe_extract_features(index_url):
        i, url = index_url
        extractor = URLFeatureExtractor(url)
        features = extractor.extract_features()
        if features is None:
            return None
        return i, features

    # -------- Load balanced dataset (1500 each class) --------
    df_full = pd.read_csv("url_dataset_balanced.csv")
    df_benign = df_full[df_full['label'] == 0].sample(n=7500, random_state=42)
    df_malicious = df_full[df_full['label'] == 1].sample(n=7500, random_state=42)
    df = pd.concat([df_benign, df_malicious]).reset_index(drop=True)
    print("Label distribution (balanced):\n", df['label'].value_counts())

    # -------- Extract features concurrently --------
    print("⚙️ Extracting features...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(executor.map(safe_extract_features, enumerate(df['url'])), total=len(df)))

    valid_results = [res for res in results if res is not None]
    if not valid_results:
        raise Exception("❌ No valid URLs were processed. Check your dataset or internet connection.")

    indices, feature_rows = zip(*valid_results)
    features_df = pd.DataFrame(feature_rows)
    labels = df.loc[list(indices), 'label'].reset_index(drop=True)

    # -------- Stratified split --------
    X_train, X_temp, y_train, y_temp = train_test_split(
        features_df, labels, test_size=0.4, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # -------- Initialize XGBoost model (fast training) --------
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        gamma=1,
        min_child_weight=2,
        reg_lambda=1,
        eval_metric='logloss',
        use_label_encoder=False,
        verbosity=1,
        tree_method='hist',  # faster histogram-based training
        n_jobs=8             # parallel threads
    )

    print("🚀 Training model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )

    # -------- Evaluate --------
    y_pred = model.predict(X_test)
    print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # -------- Save model --------
    model.save_model("url_xgb_model.json")
    print("✅ Model saved as url_xgb_model.json")
