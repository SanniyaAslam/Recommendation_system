# 🎬 CineMatch — AI Movie Recommendation System

An AI-powered movie recommendation system built with Python and Streamlit, using the real **MovieLens dataset** (9,742 movies · 100,836 ratings).

![Python](https://img.shields.io/badge/Python-3.8+-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-UI-red) ![ML](https://img.shields.io/badge/ML-Scikit--Learn-orange)

---

## ✨ Features

- 🎯 **Content-Based Filtering** — Recommends movies with similar genres using TF-IDF vectorization and Cosine Similarity
- 👥 **Collaborative Filtering** — Recommends movies based on what similar users liked using Pearson Correlation
- 🖥️ **Beautiful Web UI** — Built with Streamlit, dark Netflix-inspired theme
- 📊 **Live Stats** — Shows dataset size, number of users, and AI methods used
- 🔍 **Interactive** — Pick any movie or user and get instant top-N recommendations

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | TF-IDF + Cosine Similarity |
| Streamlit | Web interface |

---

## 📁 Project Structure

```
recommendation-system/
│
├── app.py                  # Streamlit web app
├── recommendation_system.py # Core logic (terminal version)
├── movies.csv              # MovieLens movies dataset
├── ratings.csv             # MovieLens ratings dataset
├── requirements.txt        # Dependencies
└── README.md
```

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/movie-recommendation-system.git
cd movie-recommendation-system

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

---

## 🤖 How It Works

### Content-Based Filtering
1. Movie genres are converted into numerical vectors using **TF-IDF**
2. **Cosine Similarity** measures the angle between genre vectors
3. Movies with the highest similarity score are recommended

### Collaborative Filtering
1. A **User-Item matrix** is built from all ratings
2. **Pearson Correlation** finds users with similar rating patterns
3. Movies highly rated by similar users are recommended (with predicted ratings)

---

## 📊 Dataset

[MovieLens Small Dataset](https://grouplens.org/datasets/movielens/latest/) by GroupLens Research
- 9,742 movies
- 100,836 ratings
- 610 users

---

## 👩‍💻 Author

**Sanniya** — CS Student at FAST-NUCES Lahore  
Built as part of AI/ML Internship at Optimus Automate (2026)
