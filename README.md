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

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Download NLTK data:
   - Open a Python shell and run:
   ```bash
   import nltk
   nltk.download('vader_lexicon')

4. Run the application:
   ```bash
   streamlit run app.py

---

## Usage
1. Open the app in your browser (usually at http://localhost:8501/).

2. Use the sidebar to select an analysis mode:
    - **IMDb Movie Reviews:** Enter a movie name and analyze its plot.
    - **Text Analysis:** Enter any text to analyze its sentiment and emotions.
    - **Image Analysis:** Upload an image to analyze facial emotions.

3. View the results, including interactive visualizations, in the app.

---

## Tech Stack
- **Frontend:** Streamlit
- **Backend:** Python
- **APIs Used:**
    - OMDb API for fetching movie details
- **Libraries:**
    - NLP: TextBlob, VADER, Flair, text2emotion
    - Computer Vision: OpenCV, FER
    - Data Visualization: Plotly

---

## Screenshots

### Home Page

![Screenshot (25)](https://github.com/user-attachments/assets/d8323a54-9bf4-494e-9082-bae484f961b0)


### Movie Analysis

![Screenshot (18)](https://github.com/user-attachments/assets/9032e17c-81a6-4660-8fc0-cb861a29a42e)

![image](https://github.com/user-attachments/assets/4476151e-17b2-4206-8a71-5f97f27e7d32)


### Text Analysis

![Screenshot (20)](https://github.com/user-attachments/assets/6b0145dd-8cea-48d7-8bad-ea222eda3a70)

![Screenshot (21)](https://github.com/user-attachments/assets/ee0d84aa-fd4c-43e5-a608-d5aef24ea1be)

### Image Analysis

![Screenshot (22)](https://github.com/user-attachments/assets/f482a457-da84-41df-b708-9ed465a8c84e)

![Screenshot (23)](https://github.com/user-attachments/assets/8a906efd-081f-41d4-8b29-4d35e9199ffe)

![Screenshot (24)](https://github.com/user-attachments/assets/c24fb581-c99b-44e5-a336-310b234ac7a0)


---

## Future Enhancements
  - Add support for real-time video emotion analysis.
  - Expand the dataset for more nuanced sentiment analysis.

---

## Contributing
  - Contributions are welcome! Please fork the repository and submit a pull request.

---

## License  
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements  
- [Streamlit](https://streamlit.io) for the awesome framework.  
- [OMDb API](http://www.omdbapi.com/) for movie details.  
- [FER Library](https://github.com/justinshenk/fer) for facial emotion recognition.  

---

**Happy Coding! 🎉**

Replace placeholders like `<your-username>` and `<repo-name>` with your actual GitHub username and repository name. Let me know if you need further help!
