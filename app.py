import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Load Models and Preprocessing Objects ---
@st.cache_resource
def load_model_objects():
    rf_clf = joblib.load('models/random_forest.pkl')
    xgb_clf = joblib.load('models/xgboost.pkl')
    lgbm_clf = joblib.load('models/lightgbm.pkl')
    label_encoder = joblib.load('models/label_encoder.pkl')
    mood_encoder = joblib.load('models/mood_encoder.pkl')
    tfidf = joblib.load('models/tfidf_vectorizer.pkl')
    expected_columns = joblib.load('models/expected_columns.pkl')
    # Load song metadata: dataframe with columns ['Song_Name', 'Artist']
    song_metadata = pd.read_csv('models/song_metadata.csv')  # <-- Ensure this file exists
    return rf_clf, xgb_clf, lgbm_clf, label_encoder, mood_encoder, tfidf, expected_columns, song_metadata

rf_clf, xgb_clf, lgbm_clf, label_encoder, mood_encoder, tfidf, expected_columns, song_metadata = load_model_objects()

# --- Genre Dropdown Options ---
genre_names = [
    'Ambient', 'Classical', 'Funk', 'Hip-Hop', 'Pop', 'Rock'
]

# --- UI Header ---
st.title("🎵 Song Recommendation App")
st.write("Get a song recommendation based on your current mood and preferences!")

# --- Collect User Input ---
with st.form(key='input_form'):
    user_text = st.text_input("How do you feel? (Describe your mood)", value="I'm feeling happy and energetic!")
    
    # Artist free text
    user_artist = st.text_input("Who is your favorite artist?", value="Coldplay")
    # Genre dropdown
    user_genre = st.selectbox("Preferred genre?", genre_names)
    
    user_tempo = st.number_input("Tempo (BPM):", min_value=40, max_value=220, value=120, step=1)
    user_energy = st.slider("Energy (0.0 = calm, 1.0 = energetic):", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
    user_danceability = st.slider("Danceability (0.0 = low, 1.0 = high):", min_value=0.0, max_value=1.0, value=0.6, step=0.01)
    
    mood_options = mood_encoder.categories_[0].tolist()
    user_mood = st.selectbox("Select your mood:", mood_options)
    
    model_options = {
        "Random Forest": rf_clf,
        "XGBoost": xgb_clf,
        "LightGBM": lgbm_clf
    }
    model_choice = st.selectbox("Choose a model:", list(model_options.keys()))
    
    submit_button = st.form_submit_button("Recommend me a song!")

# --- Preprocessing Function ---
def preprocess_user_input(
    user_text, user_mood, user_artist, user_genre, user_tempo, user_energy, user_danceability,
    tfidf, mood_encoder, expected_columns
):
    # TF-IDF vector for user text
    tfidf_features = tfidf.transform([user_text])
    tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=[f"tfidf_{f}" for f in tfidf.get_feature_names_out()])
    
    # One-hot encode Mood
    mood_encoded = mood_encoder.transform([[user_mood]])
    mood_df = pd.DataFrame(mood_encoded, columns=mood_encoder.get_feature_names_out(['Mood']))
    
    # Identify all one-hot columns for Artist, Genre
    artist_cols = [col for col in expected_columns if col.startswith('Artist_')]
    genre_cols = [col for col in expected_columns if col.startswith('Genre_')]
    
    # Create zeroed DataFrames for Artist and Genre
    artist_onehot = pd.DataFrame(np.zeros((1, len(artist_cols))), columns=artist_cols)
    genre_onehot = pd.DataFrame(np.zeros((1, len(genre_cols))), columns=genre_cols)
    # Set the column corresponding to user input to 1, if it exists in training
    if f"Artist_{user_artist}" in artist_cols:
        artist_onehot.at[0, f"Artist_{user_artist}"] = 1
    if f"Genre_{user_genre}" in genre_cols:
        genre_onehot.at[0, f"Genre_{user_genre}"] = 1
    
    # Numeric features
    numeric_df = pd.DataFrame({
        "Tempo (BPM)": [user_tempo],
        "Energy": [user_energy],
        "Danceability": [user_danceability]
    })
    
    # Combine all
    X_input = pd.concat([numeric_df, artist_onehot, genre_onehot, mood_df, tfidf_df], axis=1)
    
    # Add any missing columns (shouldn't be needed if using correct columns, but for robustness)
    for col in expected_columns:
        if col not in X_input.columns:
            X_input[col] = 0
    # Ensure column order
    X_input = X_input[expected_columns]
    return X_input

# --- Predict and Display ---
if submit_button:
    try:
        X_input = preprocess_user_input(
            user_text, user_mood, user_artist, user_genre, user_tempo, user_energy, user_danceability,
            tfidf, mood_encoder, expected_columns
        )
        # Select model
        clf = model_options[model_choice]
        # Predict
        y_pred = clf.predict(X_input)
        song_name = label_encoder.inverse_transform([int(y_pred[0])])[0]
        # --- Get artist for the predicted song
        # Expecting song_metadata to have columns: 'Song_Name', 'Artist'
        song_row = song_metadata[song_metadata['Song_Name'] == song_name]
        if not song_row.empty:
            artist_name = song_row['Artist'].values[0]
            st.success(f"🎶 Recommended Song: {song_name} by {artist_name}")
        else:
            st.success(f"🎶 Recommended Song: {song_name} (artist unknown)")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.markdown("---")
st.caption("Powered by Streamlit, Scikit-learn, XGBoost, and LightGBM.")