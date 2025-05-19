import streamlit as st
import joblib
import numpy as np
import pandas as pd

# --- Load Models and Preprocessing Objects ---
@st.cache_resource
def load_model_objects():
    rf_clf = joblib.load('models/random_forest.pkl')
    xgb_clf = joblib.load('models/xgboost.pkl')
    lgbm_clf = joblib.load('models/lightgbm.pkl')
    label_encoder = joblib.load('models/label_encoder.pkl')
    mood_encoder = joblib.load('models/mood_encoder.pkl')
    tfidf = joblib.load('models/tfidf_vectorizer.pkl')
    return rf_clf, xgb_clf, lgbm_clf, label_encoder, mood_encoder, tfidf

rf_clf, xgb_clf, lgbm_clf, label_encoder, mood_encoder, tfidf = load_model_objects()

# --- UI Header ---
st.title("🎵 Song Recommendation App")
st.write("Get a song recommendation based on your current mood!")

# --- User Input ---
user_text = st.text_input("How do you feel? (Describe your mood)", value="I'm feeling happy and energetic!")
# Example mood options (replace with your dataset's actual moods if needed)
mood_options = mood_encoder.categories_[0].tolist()
user_mood = st.selectbox("Select your mood:", mood_options)

model_options = {
    "Random Forest": rf_clf,
    "XGBoost": xgb_clf,
    "LightGBM": lgbm_clf
}
model_choice = st.selectbox("Choose a model:", list(model_options.keys()))

# --- Preprocess Input for Prediction ---
def preprocess_user_input(user_text, user_mood, tfidf, mood_encoder):
    # TF-IDF vector for user text
    tfidf_features = tfidf.transform([user_text])
    tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=[f"tfidf_{f}" for f in tfidf.get_feature_names_out()])
    # One-hot for mood
    mood_encoded = mood_encoder.transform([[user_mood]])
    mood_df = pd.DataFrame(mood_encoded, columns=mood_encoder.get_feature_names_out(['Mood']))
    # Combine features (order must match training!)
    X_input = pd.concat([mood_df, tfidf_df], axis=1)
    return X_input

# --- Predict and Display ---
if st.button("Recommend me a song!"):
    X_input = preprocess_user_input(user_text, user_mood, tfidf, mood_encoder)
    # Select model
    clf = model_options[model_choice]
    # Predict
    y_pred = clf.predict(X_input)
    song_name = label_encoder.inverse_transform([int(y_pred[0])])[0]
    st.success(f"🎶 Recommended Song: **{song_name}**")

st.markdown("---")
st.caption("Powered by Streamlit, Scikit-learn, XGBoost, and LightGBM.")