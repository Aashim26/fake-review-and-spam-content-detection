# app.py — Fake Review Detection Pro (Stylish + Graphs + Spam Override)
# ---------------------------------------------------------------
# Requirements:
#   pip install streamlit pandas scikit-learn joblib numpy matplotlib
#
# Run:
#   streamlit run app.py
# ---------------------------------------------------------------

import os
import json
import re
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt

MODEL_DIR = "models"


# ---------------------- Helpers ----------------------
def load_artifacts():
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "text_vectorizer.pkl"))
    text_model = joblib.load(os.path.join(MODEL_DIR, "text_model.pkl"))
    behavior_model = joblib.load(os.path.join(MODEL_DIR, "behavior_model.pkl"))
    feature_cols = json.load(open(os.path.join(MODEL_DIR, "feature_cols.json"), "r", encoding="utf-8"))
    meta = json.load(open(os.path.join(MODEL_DIR, "meta.json"), "r", encoding="utf-8"))
    return vectorizer, text_model, behavior_model, feature_cols, meta


def safe_num(x, default=0.0):
    try:
        v = float(x)
        if np.isfinite(v):
            return v
        return default
    except:
        return default


def compute_behavior_vector(user_inputs: dict, feature_cols: list) -> pd.DataFrame:
    row = {c: safe_num(user_inputs.get(c, 0.0), 0.0) for c in feature_cols}

    # Derived features (only if model expects them)
    if "rating_minus_restaurant" in feature_cols:
        row["rating_minus_restaurant"] = safe_num(user_inputs.get("rating", 0)) - safe_num(user_inputs.get("restaurantRating", 0))

    if "useful_per_reviewcount" in feature_cols:
        rc = max(safe_num(user_inputs.get("reviewCount", 0)), 1.0)
        row["useful_per_reviewcount"] = safe_num(user_inputs.get("reviewUsefulCount", 0)) / rc

    if "engagement_sum" in feature_cols:
        s = 0.0
        for c in ["usefulCount", "coolCount", "funnyCount", "complimentCount", "tipCount", "fanCount", "friendCount"]:
            if c in feature_cols:
                s += safe_num(user_inputs.get(c, 0))
        row["engagement_sum"] = s

    return pd.DataFrame([row], columns=feature_cols)


def score_to_label(p_fake: float, threshold: float) -> str:
    return "Fake Review ❌" if p_fake >= threshold else "Genuine Review ✅"


def is_spam_review(text: str) -> tuple[bool, int, list]:
    """
    Rule-based spam detector (since dataset label is fake/genuine).
    Returns (is_spam, spam_score, matched_signals)
    """
    t = (text or "").lower()
    signals = []
    score = 0

    spam_keywords = [
        "free", "coupon", "cashback", "offer", "discount", "deal", "promo", "limited time",
        "refund", "guarantee", "click", "link", "whatsapp", "call", "dm", "telegram", "contact",
        "subscribe", "join", "winner", "gift", "prize", "affiliate"
    ]

    hits = [k for k in spam_keywords if k in t]
    if hits:
        score += min(len(hits), 5)
        signals.append(f"Spam keywords found: {', '.join(hits[:6])}{'...' if len(hits) > 6 else ''}")

    if "http://" in t or "https://" in t or "www." in t:
        score += 2
        signals.append("Contains URL/link.")

    # Loose phone number detection (10 digits)
    if re.search(r"\b\d{10}\b", t):
        score += 2
        signals.append("Contains phone number pattern (10 digits).")

    if t.count("!") >= 3:
        score += 1
        signals.append("Excessive exclamation marks.")

    return (score >= 3), score, signals


def rule_signals(review: str, rating: float, restaurant_rating: float) -> list[str]:
    signals = []
    words = (review or "").split()

    if len(words) <= 3:
        signals.append("Very short / generic review text.")
    if "!!!" in review or review.count("!") >= 3:
        signals.append("Excessive exclamation marks.")
    if review.isupper() and len(review) > 20:
        signals.append("ALL CAPS writing style.")
    if abs((rating - restaurant_rating)) >= 2:
        signals.append("Rating deviates strongly from restaurant average.")
    if any(k in review.lower() for k in ["refund", "free", "offer", "click", "whatsapp", "link"]):
        signals.append("Contains promotional/spam-like terms.")
    if not signals:
        signals = ["No strong rule-based red flags detected. Decision mainly from learned patterns."]
    return signals


def plot_confidence_bar(fake_pct: float, gen_pct: float):
    fig = plt.figure()
    labels = ["Genuine", "Fake"]
    values = [gen_pct, fake_pct]
    plt.bar(labels, values)
    plt.ylim(0, 100)
    plt.title("Prediction Confidence (%)")
    plt.ylabel("Confidence")
    return fig


def plot_component_bar(p_text: float, p_beh: float, p_hyb: float):
    fig = plt.figure()
    labels = ["Text Model", "Behavior Model", "Hybrid"]
    values = [p_text * 100, p_beh * 100, p_hyb * 100]
    plt.bar(labels, values)
    plt.ylim(0, 100)
    plt.title("Fraud Probability by Component (%)")
    plt.ylabel("Probability")
    return fig


# ---------------------- Page config ----------------------
st.set_page_config(page_title="Fake Review Detection Pro", page_icon="🛡️", layout="wide")

# ---------------------- Stylish theme (CSS) ----------------------
st.markdown(
    """
    <style>
      .hero {
        padding: 18px 18px;
        border-radius: 22px;
        background:
          radial-gradient(circle at 20% 0%, rgba(99,102,241,0.34), transparent 38%),
          radial-gradient(circle at 85% 20%, rgba(34,197,94,0.24), transparent 40%),
          radial-gradient(circle at 50% 120%, rgba(236,72,153,0.18), transparent 40%),
          linear-gradient(135deg, rgba(15,23,42,0.92), rgba(2,6,23,0.92));
        border: 1px solid rgba(255,255,255,0.10);
      }
      .hero h1 { margin: 0; font-size: 34px; letter-spacing: 0.2px; }
      .hero p { margin: 6px 0 0 0; opacity: 0.92; }

      .pill {
        display:inline-flex;
        align-items:center;
        gap:6px;
        padding: 6px 10px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.14);
        background: rgba(255,255,255,0.06);
        font-size: 12px;
        margin-right: 6px;
        color: rgba(255,255,255,0.92);
      }

      .glass {
        padding: 14px 14px;
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.06);
      }

      .hint {
        border-radius: 16px;
        padding: 10px 12px;
        border: 1px dashed rgba(255,255,255,0.18);
        background: rgba(255,255,255,0.05);
      }

      .metric-row {
        display:flex;
        gap:10px;
        flex-wrap:wrap;
      }
      .metric {
        flex: 1 1 140px;
        padding: 12px 12px;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.12);
        background: rgba(255,255,255,0.06);
      }
      .metric .k { font-size: 12px; opacity:0.80; }
      .metric .v { font-size: 20px; font-weight:700; margin-top:2px; }

      .barwrap { width:100%; background: rgba(255,255,255,0.12); border-radius: 999px; overflow:hidden; height: 12px; }
      .barfill { height: 12px; border-radius: 999px; }

      .subtle { opacity: 0.82; }
      .small { font-size: 12px; opacity: 0.80; }
      .section-title { margin-top: 6px; }

      /* Make buttons nicer */
      div.stButton > button {
        border-radius: 14px;
        padding: 10px 12px;
      }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------- Header ----------------------
st.markdown(
    """
    <div class="hero">
      <span class="pill">🧠 Hybrid AI</span>
      <span class="pill">📝 Text + Behavior</span>
      <span class="pill">📊 Risk Scoring</span>
      <span class="pill">📦 Batch Page</span>
      <h1>🛡️ Fake Review Detection Pro</h1>
      <p>Professional fake review detection using linguistic signals + reviewer behavior metadata, with confidence graphs.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar help
st.sidebar.markdown("### Navigation")
st.sidebar.write("• **Main Page**: Single review analysis")
st.sidebar.write("• **Batch Predict**: Pages ➜ Batch Predict (CSV upload & download)")
st.sidebar.markdown("---")
st.sidebar.markdown("### Quick Tips")
st.sidebar.write("• Increase **threshold** (0.65–0.75) to reduce false positives.")
st.sidebar.write("• Add reviewer metadata for better accuracy (behavior model is strong).")

# ---------------------- Load models ----------------------
if not os.path.exists(MODEL_DIR):
    st.error("Models folder not found. Please run `python train.py` first.")
    st.stop()

vectorizer, text_model, behavior_model, feature_cols, meta = load_artifacts()
w_text = meta.get("weights", {}).get("text", 0.6)
w_beh = meta.get("weights", {}).get("behavior", 0.4)

# ---------------------- Examples ----------------------
EXAMPLES = {
    "✅ Real Example": {
        "text": "Visited with family. The biryani was flavorful and the staff were polite. Waiting time was a bit high but overall a good experience.",
        "rating": 4, "restaurantRating": 4.1, "ReviewLength": 26,
        "reviewUsefulCount": 2, "reviewCount": 180, "friendCount": 35,
        "usefulCount": 120, "coolCount": 40, "funnyCount": 10,
        "complimentCount": 8, "tipCount": 5, "fanCount": 2,
    },
    "❌ Fake Example": {
        "text": "BEST PLACE EVER!!! AMAZING FOOD!!! MUST VISIT!!!",
        "rating": 5, "restaurantRating": 3.4, "ReviewLength": 8,
        "reviewUsefulCount": 0, "reviewCount": 1, "friendCount": 0,
        "usefulCount": 0, "coolCount": 0, "funnyCount": 0,
        "complimentCount": 0, "tipCount": 0, "fanCount": 0,
    },
    "🚫 Spam Example": {
        "text": "Limited time offer! Click the link to get FREE coupon and cashback. WhatsApp us now for refund guarantee!",
        "rating": 5, "restaurantRating": 4.2, "ReviewLength": 22,
        "reviewUsefulCount": 0, "reviewCount": 3, "friendCount": 0,
        "usefulCount": 0, "coolCount": 0, "funnyCount": 0,
        "complimentCount": 0, "tipCount": 0, "fanCount": 0,
    }
}

if "review_text" not in st.session_state:
    st.session_state.review_text = ""
if "example_payload" not in st.session_state:
    st.session_state.example_payload = {}

# ---------------------- Layout ----------------------
left, right = st.columns([1.08, 0.92], gap="large")

with left:
    st.markdown("### 🧪 Quick Examples")
    b1, b2, b3 = st.columns(3)
    if b1.button("✅ Real", use_container_width=True):
        st.session_state.review_text = EXAMPLES["✅ Real Example"]["text"]
        st.session_state.example_payload = EXAMPLES["✅ Real Example"]
    if b2.button("❌ Fake", use_container_width=True):
        st.session_state.review_text = EXAMPLES["❌ Fake Example"]["text"]
        st.session_state.example_payload = EXAMPLES["❌ Fake Example"]
    if b3.button("🚫 Spam", use_container_width=True):
        st.session_state.review_text = EXAMPLES["🚫 Spam Example"]["text"]
        st.session_state.example_payload = EXAMPLES["🚫 Spam Example"]

    st.markdown("### ✍️ Review Text")
    review = st.text_area(
        "Paste a customer review",
        height=170,
        value=st.session_state.review_text,
        placeholder="Example: The food was amazing, service was quick, and ambience was great..."
    )
    st.session_state.review_text = review

    st.markdown("### 👤 Reviewer & Context (Optional but improves accuracy)")
    ex = st.session_state.example_payload or {}
    exv = lambda k, d: ex.get(k, d)

    tab1, tab2 = st.tabs(["Core Inputs", "Engagement Inputs"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        with c1:
            rating = st.slider("Rating", 1, 5, int(exv("rating", 4)))
            restaurantRating = st.slider("Restaurant Avg Rating", 1.0, 5.0, float(exv("restaurantRating", 4.0)), 0.1)
        with c2:
            ReviewLength = st.number_input("Review Length (words)", min_value=0, value=int(exv("ReviewLength", 60)))
            reviewUsefulCount = st.number_input("Review Useful Count", min_value=0, value=int(exv("reviewUsefulCount", 0)))
        with c3:
            reviewCount = st.number_input("Reviewer Total Reviews", min_value=0, value=int(exv("reviewCount", 10)))
            friendCount = st.number_input("Friend Count", min_value=0, value=int(exv("friendCount", 0)))

    with tab2:
        c4, c5, c6 = st.columns(3)
        with c4:
            usefulCount = st.number_input("Useful Votes (overall)", min_value=0, value=int(exv("usefulCount", 0)))
            complimentCount = st.number_input("Compliment Count", min_value=0, value=int(exv("complimentCount", 0)))
        with c5:
            coolCount = st.number_input("Cool Votes", min_value=0, value=int(exv("coolCount", 0)))
            tipCount = st.number_input("Tip Count", min_value=0, value=int(exv("tipCount", 0)))
        with c6:
            funnyCount = st.number_input("Funny Votes", min_value=0, value=int(exv("funnyCount", 0)))
            fanCount = st.number_input("Fan Count", min_value=0, value=int(exv("fanCount", 0)))

    st.markdown("### ⚙️ Decision Controls")
    threshold = st.slider(
        "Fake threshold (higher = fewer genuine flagged)",
        min_value=0.50,
        max_value=0.90,
        value=0.65,
        step=0.01
    )

    st.markdown(
        "<div class='hint'>📌 Tip: The <b>Batch Prediction</b> page is available in the left sidebar (Pages ➜ Batch Predict). You can upload a CSV file and download the prediction results.</div>",
        unsafe_allow_html=True
    )

    predict_btn = st.button("🔎 Analyze Review", use_container_width=True)

with right:
    st.markdown("### 📌 Result")
    result_box = st.empty()

    st.markdown("### 📊 Confidence")
    conf_box = st.empty()

    st.markdown("### 📈 Confidence Graph")
    graph_box = st.empty()

    st.markdown("### 🧩 Component Graph")
    comp_graph_box = st.empty()

    st.markdown("### 🧠 Why it looks suspicious")
    explain_box = st.empty()

# ---------------------- Prediction ----------------------
def progress_cards(fake_pct: float, gen_pct: float) -> str:
    fake_pct = float(np.clip(fake_pct, 0, 100))
    gen_pct = float(np.clip(gen_pct, 0, 100))

    return f"""
      <div class="glass">
        <div class="metric-row">
          <div class="metric">
            <div class="k">Fake Confidence</div>
            <div class="v" style="color: rgba(248,113,113,0.95);">{fake_pct:.2f}%</div>
            <div class="barwrap"><div class="barfill" style="width:{fake_pct}%; background: rgba(239,68,68,0.85);"></div></div>
          </div>
          <div class="metric">
            <div class="k">Genuine Confidence</div>
            <div class="v" style="color: rgba(74,222,128,0.95);">{gen_pct:.2f}%</div>
            <div class="barwrap"><div class="barfill" style="width:{gen_pct}%; background: rgba(34,197,94,0.85);"></div></div>
          </div>
        </div>
        <div class="small subtle" style="margin-top:10px;">
          Tip: Increase the threshold to reduce false positives (genuine flagged as fake).
        </div>
      </div>
    """

if predict_btn:
    if not (review or "").strip():
        st.warning("Review text is empty. Please paste a review.")
        st.stop()

    # --- Text probability ---
    X_text = vectorizer.transform([review.lower().strip()])
    p_text = float(text_model.predict_proba(X_text)[:, 1][0])

    # --- Behavior probability ---
    user_inputs = {
        "rating": rating,
        "restaurantRating": restaurantRating,
        "ReviewLength": ReviewLength,
        "reviewUsefulCount": reviewUsefulCount,
        "reviewCount": reviewCount,
        "friendCount": friendCount,
        "usefulCount": usefulCount,
        "coolCount": coolCount,
        "funnyCount": funnyCount,
        "complimentCount": complimentCount,
        "tipCount": tipCount,
        "fanCount": fanCount,
    }
    Xb = compute_behavior_vector(user_inputs, feature_cols)
    p_beh = float(behavior_model.predict_proba(Xb)[:, 1][0])

    # --- Hybrid ---
    p_hybrid = (w_text * p_text) + (w_beh * p_beh)

    # --- Spam override ---
    spam_flag, spam_score, spam_signals = is_spam_review(review)

    if spam_flag:
        label = "Spam Review 🚫"
        # For display, treat spam as high-risk
        p_display = max(p_hybrid, 0.85)
    else:
        label = score_to_label(p_hybrid, threshold=threshold)
        p_display = p_hybrid

    fake_pct = p_display * 100
    gen_pct = (1 - p_display) * 100

    # Result card
    result_box.markdown(
        f"""
        <div class="glass">
          <span class="pill">Text weight: {w_text:.2f}</span>
          <span class="pill">Behavior weight: {w_beh:.2f}</span>
          <span class="pill">Threshold: {threshold:.2f}</span>
          <h2 class="section-title" style="margin:10px 0 0 0;">{label}</h2>
          <p class="subtle" style="margin:6px 0 0 0;">
            Hybrid fraud probability: <b>{p_hybrid*100:.2f}%</b>
            {f" • Spam score: <b>{spam_score}</b>" if spam_flag else ""}
          </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Confidence bars
    conf_box.markdown(progress_cards(fake_pct, gen_pct), unsafe_allow_html=True)

    # Graphs
    fig1 = plot_confidence_bar(fake_pct=fake_pct, gen_pct=gen_pct)
    graph_box.pyplot(fig1, clear_figure=True)

    fig2 = plot_component_bar(p_text=p_text, p_beh=p_beh, p_hyb=p_hybrid)
    comp_graph_box.pyplot(fig2, clear_figure=True)

    # Explain signals
    signals = []
    signals.extend(rule_signals(review, float(rating), float(restaurantRating)))
    if spam_flag and spam_signals:
        signals = ["⚠️ Spam override triggered."] + spam_signals + signals

    explain_box.markdown(
        "<div class='glass'><ul>" + "".join([f"<li>{s}</li>" for s in signals]) + "</ul></div>",
        unsafe_allow_html=True
    )

# Footer
st.markdown("---")
st.caption("For real deployment: keep a higher threshold + a manual review queue for borderline cases.")