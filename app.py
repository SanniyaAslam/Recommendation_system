import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="🎬 Movie Recommender", page_icon="🎬", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0f0f1a; color: #ffffff; }
    .main-title {
        font-size: 3rem; font-weight: 800; text-align: center;
        background: linear-gradient(90deg, #e50914, #ff6b6b, #ffd700);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title { text-align: center; color: #aaaaaa; font-size: 1rem; margin-bottom: 2rem; }
    .movie-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #e50914; border-radius: 12px;
        padding: 16px 20px; margin-bottom: 12px;
    }
    .rank-badge {
        background: #e50914; color: white; border-radius: 50%;
        width: 32px; height: 32px; display: inline-flex;
        align-items: center; justify-content: center;
        font-weight: bold; font-size: 0.9rem; margin-right: 10px;
    }
    .movie-title-text { font-size: 1.05rem; font-weight: 600; color: #ffffff; }
    .genre-tag {
        display: inline-block; background: #2a2a3e; color: #ffd700;
        border-radius: 20px; padding: 2px 10px; font-size: 0.75rem;
        margin: 3px 2px; border: 1px solid #ffd70055;
    }
    .score-text { color: #e50914; font-weight: bold; font-size: 0.85rem; }
    .section-header {
        font-size: 1.3rem; font-weight: 700; color: #ffd700;
        border-left: 4px solid #e50914; padding-left: 10px;
        margin: 1.5rem 0 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #e50914, #ff6b6b);
        color: white; border: none; border-radius: 8px;
        padding: 0.6rem 2rem; font-size: 1rem; font-weight: 600; width: 100%;
    }
    .stat-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #333355; border-radius: 10px;
        padding: 16px; text-align: center;
    }
    .stat-number { font-size: 1.8rem; font-weight: 800; color: #ffd700; }
    .stat-label  { font-size: 0.85rem; color: #aaaaaa; }
    .stTabs [data-baseweb="tab"] { color: #aaaaaa; font-weight: 600; }
    .stTabs [aria-selected="true"] { color: #e50914 !important; border-bottom: 2px solid #e50914 !important; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
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
def build_user_item(_ratings_df):
    # Only keep users with at least 50 ratings to reduce matrix size
    active_users = _ratings_df.groupby("userId").filter(lambda x: len(x) >= 50)
    return active_users.pivot_table(index="userId", columns="movieId", values="rating")

@st.cache_data
def precompute_user_similarity(_user_item):
    # Precompute full similarity matrix using sklearn (much faster than loop)
    filled = _user_item.fillna(0).values
    sim    = cosine_similarity(filled)
    return pd.DataFrame(sim, index=_user_item.index, columns=_user_item.index)


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


def collab_recommend(user_id, user_item, sim_df, movies_df, top_n=5):
    if user_id not in user_item.index:
        return pd.DataFrame()

    # Get top 20 similar users instantly from precomputed matrix
    sim_scores   = sim_df[user_id].drop(user_id).sort_values(ascending=False).head(20)
    already_seen = set(user_item.loc[user_id].dropna().index)
    scores       = {}

    for sim_uid, corr in sim_scores.items():
        if corr <= 0:
            continue
        sim_ratings = user_item.loc[sim_uid].dropna()
        for mid, rating in sim_ratings.items():
            if mid not in already_seen:
                if mid not in scores:
                    scores[mid] = {"ws": 0, "cs": 0}
                scores[mid]["ws"] += corr * rating
                scores[mid]["cs"] += abs(corr)

    if not scores:
        return pd.DataFrame()

    predictions = {
        mid: round(val["ws"] / val["cs"], 2)
        for mid, val in scores.items() if val["cs"] > 0
    }
    top_movies = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:top_n]

    results = []
    for mid, pred in top_movies:
        row = movies_df[movies_df["movieId"] == mid]
        if not row.empty:
            results.append({
                "title":  row.iloc[0]["title"],
                "genres": row.iloc[0]["genres"],
                "score":  pred
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
movies, ratings     = load_data()
cosine_sim, indices = build_content_model(movies)
user_item           = build_user_item(ratings)
sim_df              = precompute_user_similarity(user_item)

st.markdown('<div class="main-title">🎬 CineMatch</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-Powered Movie Recommendation System · MovieLens Dataset</div>', unsafe_allow_html=True)

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
    st.markdown('<div class="section-header">Discover what users like you enjoyed</div>', unsafe_allow_html=True)
    st.markdown("Uses **Cosine Similarity** on user rating patterns to predict movies you'll love.")
    col1, col2 = st.columns([3, 1])
    with col1:
        user_ids      = sorted(user_item.index.tolist())
        selected_user = st.selectbox("👤 Select a User ID", user_ids, key="cf_user")
    with col2:
        top_n_cf = st.number_input("Top N", min_value=1, max_value=20, value=5, key="cf_n")
    with st.expander("📋 See this user's rated movies"):
        user_rated = ratings[ratings["userId"] == selected_user].merge(
            movies, on="movieId")[["title", "genres", "rating"]].sort_values("rating", ascending=False)
        st.dataframe(user_rated.reset_index(drop=True), use_container_width=True)
    if st.button("🔍 Get Recommendations", key="cf_btn"):
        with st.spinner("Finding recommendations..."):
            result = collab_recommend(selected_user, user_item, sim_df, movies, top_n=top_n_cf)
        st.markdown(f'<div class="section-header">Recommended for User #{selected_user}</div>', unsafe_allow_html=True)
        render_cards(result, score_label="Predicted Score", score_suffix="/5")