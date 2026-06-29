import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="🎬 MovieHub", page_icon="🎬", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .stApp { background-color: #0F172A; color: #FFFFFF; }

    .main-title {
        font-family: 'Poppins', sans-serif;
        font-size: 3.2rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #3B82F6, #38BDF8, #FFFFFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        letter-spacing: -1px;
    }
    .sub-title {
        text-align: center;
        color: #94A3B8;
        font-size: 1rem;
        margin-bottom: 2rem;
        font-family: 'Inter', sans-serif;
    }

    .movie-card {
        background: #1E293B;
        border: 1px solid #3B82F6;
        border-radius: 14px;
        padding: 18px 22px;
        margin-bottom: 12px;
        transition: transform 0.2s, border-color 0.2s;
    }
    .movie-card:hover {
        transform: scale(1.01);
        border-color: #38BDF8;
    }
    .rank-badge {
        background: linear-gradient(135deg, #3B82F6, #38BDF8);
        color: white;
        border-radius: 50%;
        width: 34px;
        height: 34px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.9rem;
        margin-right: 10px;
        font-family: 'Poppins', sans-serif;
    }
    .movie-title-text {
        font-size: 1.05rem;
        font-weight: 600;
        color: #FFFFFF;
        font-family: 'Poppins', sans-serif;
    }
    .genre-tag {
        display: inline-block;
        background: #0F172A;
        color: #38BDF8;
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 0.75rem;
        margin: 3px 2px;
        border: 1px solid #3B82F644;
        font-family: 'Inter', sans-serif;
    }
    .score-text {
        color: #38BDF8;
        font-weight: 700;
        font-size: 0.85rem;
        font-family: 'Inter', sans-serif;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #38BDF8;
        border-left: 4px solid #3B82F6;
        padding-left: 12px;
        margin: 1.5rem 0 1rem 0;
        font-family: 'Poppins', sans-serif;
    }
    .stButton > button {
        background: linear-gradient(90deg, #3B82F6, #38BDF8);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        font-family: 'Poppins', sans-serif;
        letter-spacing: 0.3px;
    }
    .stButton > button:hover { opacity: 0.9; transform: scale(1.01); }

    .stat-box {
        background: #1E293B;
        border: 1px solid #3B82F633;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
    }
    .stat-number {
        font-size: 1.9rem;
        font-weight: 800;
        color: #38BDF8;
        font-family: 'Poppins', sans-serif;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #94A3B8;
        font-family: 'Inter', sans-serif;
    }

    .stTabs [data-baseweb="tab"] {
        color: #94A3B8;
        font-weight: 600;
        font-family: 'Poppins', sans-serif;
    }
    .stTabs [aria-selected="true"] {
        color: #3B82F6 !important;
        border-bottom: 2px solid #3B82F6 !important;
    }

    .stSelectbox > div > div {
        background-color: #1E293B !important;
        color: white !important;
        border: 1px solid #3B82F644 !important;
        border-radius: 8px !important;
    }
    .stNumberInput > div > div > input {
        background-color: #1E293B !important;
        color: white !important;
    }

    div[data-testid="stExpander"] {
        background: #1E293B;
        border: 1px solid #3B82F633;
        border-radius: 10px;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── DATA LOADING ──────────────────────────────────────────
@st.cache_data
def load_data():
    movies  = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    return movies, ratings

@st.cache_data
def build_content_model(movies_df):
    df = movies_df.copy()
    df["genres_clean"] = df["genres"].str.replace("|", " ", regex=False)
    tfidf      = TfidfVectorizer()
    tfidf_mat  = tfidf.fit_transform(df["genres_clean"])
    cosine_sim = cosine_similarity(tfidf_mat, tfidf_mat)
    indices    = pd.Series(df.index, index=df["title"]).drop_duplicates()
    return cosine_sim, indices

@st.cache_data
def prepare_collab_data(_ratings_df):
    movie_stats = _ratings_df.groupby("movieId").agg(
        avg_rating=("rating", "mean"),
        num_ratings=("rating", "count")
    ).reset_index()
    movie_stats = movie_stats[movie_stats["num_ratings"] >= 10]
    user_ratings = _ratings_df.groupby("userId").apply(
        lambda x: dict(zip(x["movieId"], x["rating"]))
    ).to_dict()
    return movie_stats, user_ratings


# ── RECOMMENDATION FUNCTIONS ──────────────────────────────
def content_recommend(movie_title, movies_df, cosine_sim, indices, top_n=5):
    if movie_title not in indices:
        return pd.DataFrame()
    idx        = indices[movie_title]
    sim_scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1: top_n + 1]
    movie_idx  = [i[0] for i in sim_scores]
    scores     = [round(i[1], 4) for i in sim_scores]
    result     = movies_df.iloc[movie_idx][["title", "genres"]].copy()
    result["score"] = scores
    return result.reset_index(drop=True)


def collab_recommend(user_id, user_ratings, movie_stats, movies_df, ratings_df, top_n=5):
    if user_id not in user_ratings:
        return pd.DataFrame()
    seen_movies       = set(user_ratings[user_id].keys())
    user_ratings_list = user_ratings[user_id]
    high_rated_ids    = [mid for mid, r in user_ratings_list.items() if r >= 4.0]
    fav_genres = set()
    for mid in high_rated_ids:
        row = movies_df[movies_df["movieId"] == mid]
        if not row.empty:
            for g in row.iloc[0]["genres"].split("|"):
                fav_genres.add(g)

    unseen = movie_stats[~movie_stats["movieId"].isin(seen_movies)].copy()
    unseen["norm_rating"] = (unseen["avg_rating"] - 1) / 4.0

    def genre_bonus(movie_id):
        row = movies_df[movies_df["movieId"] == movie_id]
        if row.empty:
            return 0
        genres = set(row.iloc[0]["genres"].split("|"))
        return 0.2 if genres & fav_genres else 0

    unseen["genre_boost"] = unseen["movieId"].apply(genre_bonus)
    unseen["final_score"]  = (unseen["norm_rating"] + unseen["genre_boost"]).round(4)
    unseen = unseen.sort_values("final_score", ascending=False).head(top_n)

    results = []
    for _, row in unseen.iterrows():
        movie_row = movies_df[movies_df["movieId"] == row["movieId"]]
        if not movie_row.empty:
            results.append({
                "title":  movie_row.iloc[0]["title"],
                "genres": movie_row.iloc[0]["genres"],
                "score":  round(row["avg_rating"], 2)
            })
    return pd.DataFrame(results)


def render_cards(df, score_label="Similarity", score_suffix=""):
    if df.empty:
        st.warning("No recommendations found.")
        return
    for i, row in df.iterrows():
        genres_html = "".join(
            f'<span class="genre-tag">{g.strip()}</span>'
            for g in row["genres"].split("|")
        )
        score_val     = row.get("score", "")
        score_display = f"{score_val}{score_suffix}" if score_val != "" else ""
        st.markdown(f"""
        <div class="movie-card">
            <span class="rank-badge">#{i+1}</span>
            <span class="movie-title-text">{row['title']}</span>
            <span class="score-text" style="float:right">{score_label}: {score_display}</span>
            <br><br>{genres_html}
        </div>
        """, unsafe_allow_html=True)


# ── MAIN APP ──────────────────────────────────────────────
movies, ratings                = load_data()
cosine_sim, indices            = build_content_model(movies)
movie_stats, user_ratings_dict = prepare_collab_data(ratings)

st.markdown('<div class="main-title">🎬 MovieHub</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-Powered Movie Recommendation System &nbsp;·&nbsp; MovieLens Dataset</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="stat-box"><div class="stat-number">{len(movies):,}</div><div class="stat-label">Movies</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="stat-box"><div class="stat-number">{len(ratings):,}</div><div class="stat-label">Ratings</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="stat-box"><div class="stat-number">{ratings["userId"].nunique():,}</div><div class="stat-label">Users</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="stat-box"><div class="stat-number">2</div><div class="stat-label">AI Methods</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
tab1, tab2 = st.tabs(["🎯  Content-Based Filtering", "👥  Collaborative Filtering"])

with tab1:
    st.markdown('<div class="section-header">Find movies similar to one you love</div>', unsafe_allow_html=True)
    st.markdown("Uses **TF-IDF + Cosine Similarity** on movie genres to find the closest matches.")
    col1, col2 = st.columns([3, 1])
    with col1:
        movie_list     = sorted(movies["title"].tolist())
        selected_movie = st.selectbox("🎬 Select a Movie", movie_list,
            index=movie_list.index("Matrix, The (1999)") if "Matrix, The (1999)" in movie_list else 0)
    with col2:
        top_n_cb = st.number_input("Top N", min_value=1, max_value=20, value=5, key="cb_n")
    if st.button("🔍 Get Recommendations", key="cb_btn"):
        with st.spinner("Finding similar movies..."):
            result = content_recommend(selected_movie, movies, cosine_sim, indices, top_n=top_n_cb)
        st.markdown(f'<div class="section-header">Because you liked: {selected_movie}</div>', unsafe_allow_html=True)
        render_cards(result, score_label="Cosine Score")

with tab2:
    st.markdown('<div class="section-header">Discover movies matched to your taste</div>', unsafe_allow_html=True)
    st.markdown("Recommends unseen movies based on **your rating history** and **genre preferences**.")
    col1, col2 = st.columns([3, 1])
    with col1:
        user_ids      = sorted(user_ratings_dict.keys())
        selected_user = st.selectbox("👤 Select a User ID", user_ids, key="cf_user")
    with col2:
        top_n_cf = st.number_input("Top N", min_value=1, max_value=20, value=5, key="cf_n")
    with st.expander("📋 See this user's rated movies"):
        user_rated = ratings[ratings["userId"] == selected_user].merge(
            movies, on="movieId")[["title", "genres", "rating"]].sort_values("rating", ascending=False)
        st.dataframe(user_rated.reset_index(drop=True), use_container_width=True)
    if st.button("🔍 Get Recommendations", key="cf_btn"):
        with st.spinner("Finding recommendations..."):
            result = collab_recommend(selected_user, user_ratings_dict, movie_stats, movies, ratings, top_n=top_n_cf)
        st.markdown(f'<div class="section-header">Recommended for User #{selected_user}</div>', unsafe_allow_html=True)
        render_cards(result, score_label="Avg Rating", score_suffix="/5")