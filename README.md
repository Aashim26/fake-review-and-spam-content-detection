# 🛡️ Fake Review Detection Pro

A Hybrid Machine Learning system that detects **Fake**, **Genuine**, and **Spam** reviews using both linguistic analysis and reviewer behavioral patterns.

---

## 📌 Project Overview

Fake reviews can mislead customers, manipulate ratings, and damage trust in online platforms.  
This project builds a hybrid AI-based fraud detection system that analyzes:

- 📝 Review text (What is written)
- 👤 Reviewer behavior (How the user behaves)

The system combines text-based classification and behavioral anomaly detection to produce a reliable final prediction with confidence scoring.

---

## 🧠 Models Used

### 1️⃣ Text Model
- TF-IDF (Feature Extraction)
- Logistic Regression (Classifier)

Detects:
- Exaggerated language
- Generic short reviews
- Spam keywords
- Suspicious writing patterns

---

### 2️⃣ Behavioral Model
- Random Forest Classifier

Analyzes:
- Review count
- Friend count
- Engagement metrics (useful, cool, funny votes)
- Rating deviation
- Activity patterns

---

### 3️⃣ Hybrid Model

Final prediction is calculated using:

Hybrid Score =  
(Text Probability × Weight) +  
(Behavior Probability × Weight)

This improves accuracy and reduces false positives.

---

## 🚀 Features

- ✅ Single review analysis
- 📊 Confidence percentage display
- 📈 Prediction graphs
- 🚫 Spam detection (rule-based override)
- 📦 Batch CSV upload & prediction
- 📥 Download prediction results
- ⚙️ Adjustable fraud threshold

---

## 📊 Model Performance

| Model       | Accuracy | AUC  |
|------------|----------|------|
| Text Model | ~65%     | 0.69 |
| Behavior   | ~89%     | 0.94 |
| Hybrid     | ~86%     | 0.93 |

Behavioral features significantly improve detection accuracy.

---

## 🏗️ Project Structure
FakeReviewPro/
│
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── models/
└── pages/
└── 1_Batch_Predict.py
