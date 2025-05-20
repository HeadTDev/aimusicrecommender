import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Spotify Style Song Recommender",
    layout="centered",
    initial_sidebar_state="auto",
)

st.markdown("""
    <style>
    body { background-color: #191414; }
    .stApp { background: linear-gradient(135deg, #191414 0%, #1DB954 100%);}
    .big-font { font-size:2.0em; font-weight: bold; color:#1DB954; }
    .spotify-btn {
        background-color: #1DB954; color: white; border: none; padding: 0.75em 2em;
        border-radius: 30px; font-weight: bold; font-size: 1.1em; margin-top: 1em;
        box-shadow: 0 4px 20px 0 rgba(30,185,84,0.25);
        transition: background 0.2s;
    }
    .spotify-btn:hover { background-color: #169643;}
    .song-card {
        background-color: #181818;
        border-radius: 16px;
        padding: 1.5em;
        margin-top: 2em;
        color: #fff;
        box-shadow: 0 2px 12px 0 rgba(30,185,84,0.08);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-font">Spotify Song Recommender 🎵</div>', unsafe_allow_html=True)
st.markdown("##### Get song suggestions and listen instantly on Spotify!")

@st.cache_data
def load_metadata():
    return pd.read_csv('models/song_metadata.csv')

song_metadata = load_metadata()

mood = st.selectbox(
    "Select your mood or type a keyword:",
    ["Happy", "Sad", "Energetic", "Chill", "Romantic", "Focus", "Workout", "Surprise Me"]
)

user_input = st.text_input("Or search for a song/artist/keyword (optional):", "")

def recommend_song(mood, user_input, df):
    mood_map = {
        "Happy": ["Happy", "Uptown Funk"],
        "Sad": ["Fix You", "Someone Like You", "Clair de Lune"],
        "Energetic": ["Stronger", "Eye of the Tiger", "Uptown Funk"],
        "Chill": ["Weightless", "Clair de Lune"],
        "Romantic": ["Someone Like You", "Clair de Lune"],
        "Focus": ["Clair de Lune", "Weightless"],
        "Workout": ["Stronger", "Eye of the Tiger", "Uptown Funk"],
        "Surprise Me": df["Song_Name"].tolist()
    }
    if user_input:
        filtered = df[df["Song_Name"].str.contains(user_input, case=False, na=False) | 
                      df["Artist"].str.contains(user_input, case=False, na=False)]
        if not filtered.empty:
            return filtered.sample(1).iloc[0]
    names = mood_map.get(mood, df["Song_Name"].tolist())
    filtered = df[df["Song_Name"].isin(names)]
    if not filtered.empty:
        return filtered.sample(1).iloc[0]
    return df.sample(1).iloc[0]

if st.button("Suggest a Song", key="suggest", help="Get a song recommendation!", 
             use_container_width=True):
    song = recommend_song(mood, user_input, song_metadata)
    song_name = song["Song_Name"]
    artist_name = song["Artist"]
    spotify_url = song.get("Spotify_URL", "")

    st.markdown(f"""
        <div class="song-card">
            <span style="font-size:1.5em; color:#1DB954;"><b>{song_name}</b></span><br>
            <span style="font-size:1.1em; color:#b3b3b3;">by {artist_name}</span><br>
        </div>
    """, unsafe_allow_html=True)

    if isinstance(spotify_url, str) and spotify_url.startswith("https://open.spotify.com/track/"):
        track_id = spotify_url.split("/")[-1].split("?")[0]
        st.markdown(
            f"""
            <div style="display:flex;justify-content:center;">
            <iframe src="https://open.spotify.com/embed/track/{track_id}"
            width="350" height="80" frameborder="0"
            allowtransparency="true" allow="encrypted-media"
            style="border-radius:12px;"></iframe>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif spotify_url:
        st.markdown(f"[Listen on Spotify]({spotify_url})")
    else:
        st.warning("No Spotify link found for this song. Try another!")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    '<div style="text-align:center; color:#b3b3b3;">Inspired by Spotify • Made with Streamlit</div>',
    unsafe_allow_html=True
)