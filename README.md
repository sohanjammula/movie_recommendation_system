# 🎬 Movie Recommendation System

A machine learning-based Movie Recommendation System that suggests similar movies based on user input using content-based filtering techniques.

---

## 📌 Overview

This project is designed to recommend movies based on similarity between movie features such as genres, keywords, cast, crew, and overview.

Recommendation systems are widely used in platforms like Netflix and Amazon to enhance user experience by providing personalized suggestions. :contentReference[oaicite:1]{index=1}

---

## 🚀 Features

- 🔍 Search for a movie by name
- 🎯 Get top similar movie recommendations
- 📊 Uses similarity metrics for accurate results
- ⚡ Fast and efficient recommendation system
- 🧠 Built using Machine Learning concepts

---

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **Libraries Used:**  
  - Pandas  
  - NumPy  
  - Scikit-learn  
  - Pickle  
- **Techniques:**  
  - NLP (text processing)  
  - CountVectorizer / TF-IDF  
  - Cosine Similarity  

---

## 📂 Project Structure
movie_recommendation_system/
│── app.py # Main application file
│── model.pkl # Trained similarity model
│── movies.pkl # Processed movie dataset
│── requirements.txt # Dependencies
│── README.md # Project documentation



---

## ⚙️ How It Works

1. Load dataset (movie metadata)
2. Perform data preprocessing:
   - Combine important features (genres, cast, etc.)
3. Convert text data into vectors using NLP
4. Calculate similarity using cosine similarity
5. Recommend movies based on highest similarity score

Content-based filtering works by comparing movie attributes and recommending similar ones. :contentReference[oaicite:2]{index=2}

---

## ▶️ Installation & Setup

```bash
git clone https://github.com/sohanjammula/movie_recommendation_system.git
cd movie_recommendation_system
pip install -r requirements.txt

python app.py
