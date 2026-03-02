# pages/1_Batch_Predict.py
import os, json
import numpy as np
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title="Batch Predict", page_icon="📦", layout="wide")

MODEL_DIR = "models"

def load_artifacts():
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "text_vectorizer.pkl"))
    text_model = joblib.load(os.path.join(MODEL_DIR, "text_model.pkl"))
    behavior_model = joblib.load(os.path.join(MODEL_DIR, "behavior_model.pkl"))
    feature_cols = json.load(open(os.path.join(MODEL_DIR, "feature_cols.json"), "r", encoding="utf-8"))
    meta = json.load(open(os.path.join(MODEL_DIR, "meta.json"), "r", encoding="utf-8"))
    return vectorizer, text_model, behavior_model, feature_cols, meta

def build_behavior_features_from_file(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    Xb = pd.DataFrame(index=df.index)

    # create all expected cols; fill missing with 0
    for col in feature_cols:
        if col in df.columns:
            Xb[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            Xb[col] = 0.0

    # derived
    if "rating_minus_restaurant" in feature_cols and "rating" in df.columns and "restaurantRating" in df.columns:
        Xb["rating_minus_restaurant"] = (
            pd.to_numeric(df["rating"], errors="coerce").fillna(0)
            - pd.to_numeric(df["restaurantRating"], errors="coerce").fillna(0)
        )

    if "useful_per_reviewcount" in feature_cols and "reviewUsefulCount" in df.columns and "reviewCount" in df.columns:
        ruc = pd.to_numeric(df["reviewUsefulCount"], errors="coerce").fillna(0)
        rc = pd.to_numeric(df["reviewCount"], errors="coerce").fillna(0).clip(lower=1)
        Xb["useful_per_reviewcount"] = ruc / rc

    if "engagement_sum" in feature_cols:
        cols = ["usefulCount", "coolCount", "funnyCount", "complimentCount", "tipCount", "fanCount", "friendCount"]
        s = 0
        for c in cols:
            if c in df.columns:
                s = s + pd.to_numeric(df[c], errors="coerce").fillna(0)
        Xb["engagement_sum"] = s

    Xb = Xb.replace([np.inf, -np.inf], np.nan).fillna(0)
    return Xb[feature_cols]

# ---- UI ----
st.markdown("## 📦 Batch Predict (CSV Upload → Predict → Download)")
st.caption("Required column: reviewContent | Optional: rating, restaurantRating, reviewCount, votes, etc.")

# Show debug info so you know it loaded
st.info("✅ Batch page loaded successfully.")

try:
    if not os.path.exists(MODEL_DIR):
        st.error("Models folder not found. Run `python train.py` first.")
        st.stop()

    vectorizer, text_model, behavior_model, feature_cols, meta = load_artifacts()
    w_text = meta.get("weights", {}).get("text", 0.6)
    w_beh  = meta.get("weights", {}).get("behavior", 0.4)

    threshold = st.slider("Fake threshold", 0.50, 0.90, 0.65, 0.01)

    up = st.file_uploader("Upload CSV", type=["csv"])

    if up is not None:
        df = pd.read_csv(up)

        if "reviewContent" not in df.columns:
            st.error("Uploaded CSV must contain column: reviewContent")
            st.stop()

        # TEXT probs
        texts = df["reviewContent"].fillna("").astype(str).str.lower().values
        X_text = vectorizer.transform(texts)
        p_text = text_model.predict_proba(X_text)[:, 1]

        # BEHAVIOR probs
        Xb = build_behavior_features_from_file(df, feature_cols)
        p_beh = behavior_model.predict_proba(Xb)[:, 1]

        # HYBRID
        p_hybrid = (w_text * p_text) + (w_beh * p_beh)

        df_out = df.copy()
        df_out["fake_probability"] = (p_hybrid * 100).round(2)
        df_out["prediction"] = np.where(p_hybrid >= threshold, "Fake", "Genuine")

        st.markdown("### Preview")
        st.dataframe(df_out.head(30), use_container_width=True)

        st.markdown("### Download")
        st.download_button(
            "⬇️ Download predictions_output.csv",
            data=df_out.to_csv(index=False).encode("utf-8"),
            file_name="predictions_output.csv",
            mime="text/csv",
            use_container_width=True
        )

except Exception as e:
    st.error("Batch page crashed. Error details below:")
    st.exception(e)