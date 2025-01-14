import streamlit as st
import requests
import json
import numpy as np
import cv2
from PIL import Image
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from flair.models import TextClassifier
from flair.data import Sentence
import text2emotion as te
import plotly.express as px
from fer import FER
import tensorflow as tf

# Suppress TensorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# OMDb API Key (replace with your actual key)
API_KEY = "890b2cf"

# Load Flair model once (instead of loading each time)
flair_classifier = TextClassifier.load('en-sentiment')

# Define the function to fetch movie details from OMDb API
def get_movie_details(query):
    url = f"http://www.omdbapi.com/?t={query}&apikey={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data
    return None

# Sentiment analysis function
def analyze_text(text):
    # TextBlob
    blob = TextBlob(text).sentiment.polarity
    
    # VADER
    vader_analyzer = SentimentIntensityAnalyzer()
    vader = vader_analyzer.polarity_scores(text)['compound']
    
    # Flair
    sentence = Sentence(text)
    flair_classifier.predict(sentence)
    flair_result = sentence.labels[0].value
    
    # Text2Emotion
    if len(text.split()) > 3:  # Only use Text2Emotion for longer text
        emotion = te.get_emotion(text)
    else:
        emotion = {'Happy': 0.0, 'Angry': 0.0, 'Surprise': 0.0, 'Sad': 0.0, 'Fear': 0.0}

    # Combine results for a more accurate analysis
    combined_sentiment = "Neutral"
    
    # Define thresholds for sentiment classification
    positive_threshold = 0.5
    negative_threshold = -0.5

    # Check TextBlob and VADER sentiments
    if blob > positive_threshold and vader > positive_threshold and flair_result == "POSITIVE":
        combined_sentiment = "Positive"
    elif blob < negative_threshold and vader < negative_threshold:
        combined_sentiment = "Negative"

    # Adjust Text2Emotion results for better handling of positive emotions
    if combined_sentiment == "Positive":
        # If the combined sentiment is positive, we can tweak the emotional weights
        if emotion['Happy'] < 0.5:  # Adjust threshold as needed
            emotion['Happy'] += 0.5  # Boost happiness if overall sentiment is positive

    return blob, vader, flair_result, emotion, combined_sentiment

# Plot emotion distribution
def plot_emotion_distribution(emotion):
    if emotion:
        fig = px.bar(x=list(emotion.keys()), y=list(emotion.values()), labels={'x': 'Emotion', 'y': 'Intensity'})
        st.plotly_chart(fig)

def analyze_image(image):
    # Convert the image to a numpy array
    img_array = np.array(image)

    # Convert the image to grayscale for face detection
    gray_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Load OpenCV's Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        st.write(f"Detected {len(faces)} face(s) in the image.")
        
        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img_array, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Display the image with detected faces
        st.image(img_array, caption="Image with Detected Faces", use_column_width=True)
        
        # Use FER for emotion detection on the original image (not grayscale)
        emotion_detector = FER()
        emotions = emotion_detector.detect_emotions(img_array)

        if emotions:
            # Extract emotions from the first detected face
            st.write("Emotion detection results:")
            st.json(emotions[0]['emotions'])

            # Plot emotion distribution for the first face
            plot_emotion_distribution(emotions[0]['emotions'])
        else:
            st.write("No emotions detected.")
    else:
        st.write("No faces detected in the image.")

# Main Streamlit app
def main():
    st.set_page_config(page_title="Sentiment Analysis App", page_icon="😊")
    
    st.sidebar.title("Sentiment Analysis App")
    option = st.sidebar.selectbox("Choose an option", ["IMDb Movie Reviews", "Text Analysis", "Image Analysis"])
    
    if option == "IMDb Movie Reviews":
        st.title("IMDb Movie Reviews")
        movie_query = st.text_input("Enter movie name:")
        
        if st.button("Search"):
            if movie_query.strip() == "":
                st.warning("Please enter a movie name.")
            else:
                with st.spinner("Fetching movie details..."):
                    movie_data = get_movie_details(movie_query)
                if movie_data and movie_data['Response'] == "True":  # Check if the movie is found
                    st.subheader(f"{movie_data['Title']} ({movie_data['Year']})")
                    st.image(movie_data['Poster'], width=200)
                    st.write(f"**Plot**: {movie_data['Plot']}")
                    st.write(f"**IMDb Rating**: {movie_data['imdbRating']}")
                    
                    # Perform sentiment analysis on movie plot
                    with st.spinner("Analyzing sentiment..."):
                        blob, vader, flair_result, emotion, combined_sentiment = analyze_text(movie_data['Plot'])
                    st.subheader("Sentiment Analysis on Plot:")
                    st.write(f"TextBlob Sentiment: {blob:.2f}")
                    st.write(f"VADER Sentiment: {vader:.2f}")
                    st.write(f"Flair Sentiment: {flair_result}")
                    st.write(f"Combined Sentiment: {combined_sentiment}")
                    
                    # Plot emotion distribution
                    st.subheader("Emotion Distribution:")
                    plot_emotion_distribution(emotion)
                else:
                    st.error("Movie not found. Please try another movie title.")
    
    elif option == "Text Analysis":
        st.title("Text Sentiment Analysis")
        text_input = st.text_area("Enter text for sentiment analysis:")
        
        if st.button("Analyze"):
            if text_input.strip() == "":
                st.warning("Please enter some text to analyze.")
            else:
                with st.spinner("Analyzing sentiment..."):
                    # Call the analyze_text function and unpack the results
                    blob, vader, flair_result, emotion, combined_sentiment = analyze_text(text_input)
                
                # Display sentiment results
                st.subheader("Sentiment Analysis Results:")
                st.write(f"TextBlob Sentiment: {blob:.2f}")
                st.write(f"VADER Sentiment: {vader:.2f}")
                st.write(f"Flair Sentiment: {flair_result}")
                st.write(f"Combined Sentiment: {combined_sentiment}")
                
                # Plot emotion distribution
                st.subheader("Emotion Distribution:")
                plot_emotion_distribution(emotion)
    
    elif option == "Image Analysis":
        st.title("Image Sentiment Analysis")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            with st.spinner("Analyzing image..."):
                analyze_image(image)

if __name__ == '__main__':
    main()
