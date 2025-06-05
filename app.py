import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and encoders
model = pickle.load(open('music_model.pkl', 'rb'))

# Load separate encoders
with open('gender_encoder.pkl', 'rb') as f:
    gender_encoder = pickle.load(f)

with open('genre_encoder.pkl', 'rb') as f:
    genre_encoder = pickle.load(f)

with open('music_encoder.pkl', 'rb') as f:
    music_encoder = pickle.load(f)

st.title("🎵 Music Like Prediction App")
st.sidebar.header("User Input Features")

def user_input():
    age = st.sidebar.slider("Age", 10, 70, 25)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    genre = st.sidebar.selectbox("Genre Preference", ['Rock', 'EDM', 'Classical', 'Country'])
    music = st.sidebar.text_input("Music Name", "Shape of You")
    minutes = st.sidebar.slider("Minutes Listened", 0, 300, 30)

    # Use individual encoders
    try:
        gender_enc = gender_encoder.transform([gender])[0]
    except:
        gender_enc = 0

    try:
        genre_enc = genre_encoder.transform([genre])[0]
    except:
        genre_enc = 0

    try:
        music_enc = music_encoder.transform([music])[0]
    except:
        music_enc = 0

    data = {
        'age': age,
        'gender': gender_enc,
        'genre_preference': genre_enc,
        'music_name': music_enc,
        'minutes_listened': minutes
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input()
st.subheader("User Input Data")
st.write(input_df)

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    result = "👍 Likely to Like the Song" if prediction == 1 else "👎 Unlikely to Like the Song"
    st.success(result)
