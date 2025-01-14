# Sentiment Analysis Web Application  

![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-orange)  
![Python](https://img.shields.io/badge/Python-3.7%2B-blue)  
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)  

## Overview  
This repository contains a **Sentiment Analysis Web Application** built using **Streamlit**. The app provides users with powerful tools to analyze sentiments and emotions from text, images, and movie reviews. It is an easy-to-use interface that leverages modern natural language processing (NLP) and computer vision techniques.  

---

## Features  
### 1. **Movie Review Analysis**  
- Fetches movie details (plot, ratings, etc.) using the **OMDb API**.  
- Performs sentiment analysis on the movie plot using multiple NLP models.  

### 2. **Text Sentiment Analysis**  
- Analyzes user-inputted text using:  
  - **TextBlob**: Sentiment polarity analysis (-1 to 1).  
  - **VADER**: Compound sentiment scores for texts.  
  - **Flair**: Deep learning-based positive/negative sentiment classification.  
- Detects emotions from text using **text2emotion**.  

### 3. **Image-Based Emotion Analysis**  
- Detects faces in uploaded images using **OpenCV**.  
- Analyzes facial emotions (e.g., happiness, anger, sadness) using the **FER library**.  

---

## Installation  

### Prerequisites  
- Python 3.7 or higher  
- A valid **OMDb API key** ([Get your API key here](http://www.omdbapi.com/apikey.aspx)).  

### Steps  
1. Clone the repository:  
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
