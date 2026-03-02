# train.py
import os, json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

DATA_PATH = "new_data_test.csv"
OUT_DIR = "models"

def robust_read_csv(path: str) -> pd.DataFrame:
    """
    Your file sometimes breaks default CSV parsing.
    This approach is more tolerant.
    """
    return pd.read_csv(path, sep=None, engine="python", on_bad_lines="skip")

def safe_to_numeric(s):
    return pd.to_numeric(s, errors="coerce")

def clean_text(s: str) -> str:
    s = "" if pd.isna(s) else str(s)
    s = s.lower()
    # keep it light; TFIDF handles tokens
    return s.strip()

def build_behavior_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make strong, real-world behavior features.
    Only uses columns if present.
    """
    out = pd.DataFrame(index=df.index)

    # Raw numeric features (if exist)
    base_cols = [
        "rating",
        "reviewUsefulCount",
        "reviewCount",
        "friendCount",
        "usefulCount",
        "coolCount",
        "funnyCount",
        "complimentCount",
        "tipCount",
        "fanCount",
        "restaurantRating",
        "ReviewLength",
    ]
    for c in base_cols:
        if c in df.columns:
            out[c] = safe_to_numeric(df[c])

    # Derived features
    # 1) rating deviation vs restaurant avg
    if "rating" in out.columns and "restaurantRating" in out.columns:
        out["rating_minus_restaurant"] = out["rating"] - out["restaurantRating"]

    # 2) helpful ratio proxy: reviewUsefulCount / max(reviewCount,1)
    if "reviewUsefulCount" in out.columns and "reviewCount" in out.columns:
        out["useful_per_reviewcount"] = out["reviewUsefulCount"] / out["reviewCount"].clip(lower=1)

    # 3) engagement totals
    eng = []
    for c in ["usefulCount", "coolCount", "funnyCount", "complimentCount", "tipCount", "fanCount", "friendCount"]:
        if c in out.columns:
            eng.append(out[c].fillna(0))
    if eng:
        out["engagement_sum"] = np.sum(np.vstack([e.values for e in eng]), axis=0)

    # Fill NaNs
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0)

    return out

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = robust_read_csv(DATA_PATH)

    # Required columns check
    if "flagged" not in df.columns:
        raise ValueError("Dataset must have 'flagged' column (0/1).")
    if "reviewContent" not in df.columns:
        raise ValueError("Dataset must have 'reviewContent' column (review text).")

    # Prepare target
    y = safe_to_numeric(df["flagged"]).fillna(0).astype(int).clip(0, 1)

    # Prepare text
    text = df["reviewContent"].apply(clean_text)

    # Behavior features
    Xb = build_behavior_features(df)
    feature_cols = Xb.columns.tolist()

    # Train/test split (same indices to align)
    idx = np.arange(len(df))
    idx_train, idx_test = train_test_split(
        idx,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ===== TEXT MODEL =====
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
        stop_words="english"
    )

    X_text_train = vectorizer.fit_transform(text.iloc[idx_train])
    X_text_test  = vectorizer.transform(text.iloc[idx_test])

    text_model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        n_jobs=None
    )
    text_model.fit(X_text_train, y.iloc[idx_train])

    text_proba_test = text_model.predict_proba(X_text_test)[:, 1]

    # ===== BEHAVIOR MODEL =====
    Xb_train = Xb.iloc[idx_train]
    Xb_test  = Xb.iloc[idx_test]

    behavior_model = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        n_jobs=-1
    )
    behavior_model.fit(Xb_train, y.iloc[idx_train])
    beh_proba_test = behavior_model.predict_proba(Xb_test)[:, 1]

    # ===== HYBRID SCORE =====
    # Weighted average – tweakable
    w_text, w_beh = 0.6, 0.4
    hybrid_proba_test = (w_text * text_proba_test) + (w_beh * beh_proba_test)

    # Metrics
    try:
        auc_text = roc_auc_score(y.iloc[idx_test], text_proba_test)
        auc_beh  = roc_auc_score(y.iloc[idx_test], beh_proba_test)
        auc_hyb  = roc_auc_score(y.iloc[idx_test], hybrid_proba_test)
    except Exception:
        auc_text = auc_beh = auc_hyb = None

    print("\n=== TEXT MODEL REPORT ===")
    print(classification_report(y.iloc[idx_test], (text_proba_test >= 0.5).astype(int)))
    print("Text AUC:", auc_text)

    print("\n=== BEHAVIOR MODEL REPORT ===")
    print(classification_report(y.iloc[idx_test], (beh_proba_test >= 0.5).astype(int)))
    print("Behavior AUC:", auc_beh)

    print("\n=== HYBRID MODEL REPORT ===")
    print(classification_report(y.iloc[idx_test], (hybrid_proba_test >= 0.5).astype(int)))
    print("Hybrid AUC:", auc_hyb)

    # Save artifacts
    joblib.dump(vectorizer, os.path.join(OUT_DIR, "text_vectorizer.pkl"))
    joblib.dump(text_model, os.path.join(OUT_DIR, "text_model.pkl"))
    joblib.dump(behavior_model, os.path.join(OUT_DIR, "behavior_model.pkl"))

    with open(os.path.join(OUT_DIR, "feature_cols.json"), "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    meta = {
        "data_path": DATA_PATH,
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "label_col": "flagged",
        "text_col": "reviewContent",
        "weights": {"text": w_text, "behavior": w_beh},
        "auc": {"text": auc_text, "behavior": auc_beh, "hybrid": auc_hyb},
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n✅ Training done. Models saved in /models")

if __name__ == "__main__":
    main()