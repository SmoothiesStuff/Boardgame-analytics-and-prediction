# streamlit_app.py — Enhanced Board Game Developer Console
# Analytics Dashboard + Improved Profile Wizard + Neighbor Analysis

import os
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from joblib import load as joblib_load

# ---------------------------------
# Page config & enhanced earth-tone theme
# ---------------------------------
st.set_page_config(page_title="Board Game Developer Console", page_icon="🎲", layout="wide")

PALETTE = [
    "#8B7355", "#A67C52", "#7C8C4A", "#C4A484", "#4F4538",
    "#B56576", "#6B705C", "#D4A373", "#8D6B94", "#9C8B7A",
    "#6B5B73", "#8A7968", "#A68B5B", "#7A8B8B", "#8B7D6B"
]
ACCENT = "#A67C52"
MUTED = "#6B705C" 
BG_SOFT = "#FAF7F2"
CHART_BG = "#FFFCF7"

st.markdown(
    f"""
    <style>
      .stApp {{ background-color: {BG_SOFT}; }}
      section[data-testid="stSidebar"] {{ background-color: #F3EFE7; }}
      h1, h2, h3, h4 {{ color: {ACCENT}; }}
      .earthcard {{ 
        border: 1px solid #e5dcc9; 
        border-radius: 12px; 
        padding: 1rem; 
        background: {CHART_BG}; 
        margin: 0.5rem 0;
      }}
      .metric-card {{
        background: {CHART_BG};
        padding: 0.75rem;
        border-radius: 8px;
        border-left: 4px solid {ACCENT};
        margin: 0.25rem 0;
      }}
      .prediction-card {{
        background: linear-gradient(135deg, {CHART_BG} 0%, #F5F1EA 100%);
        padding: 1.25rem;
        border-radius: 12px;
        border: 1px solid #E5DCC9;
        margin: 1rem 0;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------
# Enhanced configuration
# ---------------------------------
DEFAULT_PARQUET_PATH = "cleaned_large_bgg_dataset.parquet"
DEFAULT_CSV_PATH = "cleaned_large_bgg_dataset.csv"

EXCLUDE_FOR_CLUSTERING = [
    "Owned Users", "BayesAvgRating", "AvgRating", "Users Rated", "BGG Rank", 
    "StdDev", "NumWant", "NumWish", "NumComments", "NumWeightVotes",
    "ID", "BGGId", "Name", "ImagePath", "Rank:strategygames", "Rank:abstracts", 
    "Rank:familygames", "Rank:thematic", "Rank:cgs", "Rank:wargames", 
    "Rank:partygames", "Rank:childrensgames"
]

MODEL_PATHS = {
    "rating_rf": "models/rating_rf.joblib",
    "rating_xgb": "models/rating_xgb.joblib", 
    "sales_rf": "models/sales_rf.joblib",
    "sales_xgb": "models/sales_xgb.joblib",
}

CURRENT_YEAR = 2025

# ---------------------------------
# Enhanced data loading & utils
# ---------------------------------
@st.cache_data(show_spinner=True)
def load_df(uploaded_file) -> pd.DataFrame:
    try:
        if uploaded_file is not None:
            name = (getattr(uploaded_file, "name", "") or "").lower()
            if name.endswith(".parquet"):
                return pd.read_parquet(uploaded_file)
            return pd.read_csv(uploaded_file)
        if os.path.exists(DEFAULT_PARQUET_PATH):
            return pd.read_parquet(DEFAULT_PARQUET_PATH)
        if os.path.exists(DEFAULT_CSV_PATH):
            return pd.read_csv(DEFAULT_CSV_PATH)
        st.error("No dataset found. Upload a CSV/Parquet or include cleaned_large_bgg_dataset.parquet.")
        st.stop()
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        st.stop()

def split_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X = df.drop(columns=[c for c in EXCLUDE_FOR_CLUSTERING if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=["number"])
    X = X.fillna(0)
    keep_cols = [c for c in df.columns if c not in X.columns]
    if "Name" in df.columns: 
        keep_cols = list(dict.fromkeys(keep_cols + ["Name"]))
    meta = df[keep_cols].copy()
    return X, meta

@st.cache_resource(show_spinner=False)
def fit_clusterer(X: pd.DataFrame, k: int = 8, random_state: int = 42):
    if len(X) < k:
        k = max(2, len(X))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    try:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    except TypeError:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(X_scaled)
    return scaler, kmeans, pca, labels, coords

def year_percentile(series: pd.Series, value: float) -> float:
    if series is None or len(series) == 0:
        return float("nan")
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return float("nan")
    return float((s < value).mean() * 100.0)

def build_input_vector(Xcols: List[str], profile: Dict) -> pd.DataFrame:
    base = {c: 0 for c in Xcols}
    for k, v in profile.items():
        if k in base:
            base[k] = v
    return pd.DataFrame([base])[Xcols]

def nearest_neighbors_in_cluster(input_scaled, cluster_id: int, df_full: pd.DataFrame,
                                 X: pd.DataFrame, scaler, kmeans, topn: int = 10) -> pd.DataFrame:
    X_scaled_full = scaler.transform(X)
    labels = kmeans.predict(X_scaled_full)
    mask = labels == cluster_id
    Xc = X_scaled_full[mask]
    dfc = df_full.loc[X.index[mask]].copy()
    dists = pairwise_distances(input_scaled, Xc)[0]
    dfc["__dist"] = dists
    return dfc.nsmallest(topn, "__dist")

@st.cache_resource(show_spinner=False)
def load_models(paths: Dict[str, str]):
    models = {}
    for key, path in paths.items():
        if os.path.exists(path):
            try:
                models[key] = joblib_load(path)
            except Exception as e:
                st.warning(f"Could not load {key} at {path}: {e}")
    return models

def predict_with_models(models: Dict, X_input_df: pd.DataFrame) -> Dict[str, float | None]:
    out = {}
    for key, model in models.items():
        try:
            out[key] = float(model.predict(X_input_df)[0])
        except Exception:
            out[key] = None
    return out

# ---------------------------------
# Enhanced visualization functions
# ---------------------------------
def create_complexity_vs_rating_chart(df_filtered: pd.DataFrame, highlight_neighbors=None):
    """Create complexity vs rating scatter plot"""
    fig = go.Figure()
    
    # Main data points
    fig.add_trace(go.Scatter(
        x=df_filtered["GameWeight"],
        y=df_filtered["BayesAvgRating"], 
        mode='markers',
        marker=dict(size=6, color=MUTED, opacity=0.6),
        text=df_filtered["Name"],
        hovertemplate="<b>%{text}</b><br>Complexity: %{x:.1f}<br>Rating: %{y:.2f}<extra></extra>",
        name="All Games",
        showlegend=True
    ))
    
    # Highlight neighbors if provided
    if highlight_neighbors is not None and len(highlight_neighbors) > 0:
        fig.add_trace(go.Scatter(
            x=highlight_neighbors["GameWeight"],
            y=highlight_neighbors["BayesAvgRating"],
            mode='markers',
            marker=dict(size=10, color=ACCENT, opacity=0.9, line=dict(width=2, color='white')),
            text=highlight_neighbors["Name"],
            hovertemplate="<b>%{text}</b><br>Complexity: %{x:.1f}<br>Rating: %{y:.2f}<extra></extra>",
            name="Similar Games",
            showlegend=True
        ))
    
    fig.update_layout(
        title="Game Complexity vs Rating",
        xaxis_title="Complexity (GameWeight)",
        yaxis_title="Bayes Average Rating",
        plot_bgcolor=CHART_BG,
        paper_bgcolor=CHART_BG,
        height=400,
        font_color=MUTED
    )
    return fig

def create_year_vs_rating_chart(df_filtered: pd.DataFrame, highlight_neighbors=None):
    """Create year vs rating scatter plot"""
    fig = go.Figure()
    
    # Main data points
    fig.add_trace(go.Scatter(
        x=df_filtered["Year Published"],
        y=df_filtered["BayesAvgRating"],
        mode='markers',
        marker=dict(size=6, color=MUTED, opacity=0.6),
        text=df_filtered["Name"],
        hovertemplate="<b>%{text}</b><br>Year: %{x}<br>Rating: %{y:.2f}<extra></extra>",
        name="All Games",
        showlegend=True
    ))
    
    # Highlight neighbors if provided
    if highlight_neighbors is not None and len(highlight_neighbors) > 0:
        fig.add_trace(go.Scatter(
            x=highlight_neighbors["Year Published"],
            y=highlight_neighbors["BayesAvgRating"],
            mode='markers',
            marker=dict(size=10, color=ACCENT, opacity=0.9, line=dict(width=2, color='white')),
            text=highlight_neighbors["Name"],
            hovertemplate="<b>%{text}</b><br>Year: %{x}<br>Rating: %{y:.2f}<extra></extra>",
            name="Similar Games",
            showlegend=True
        ))
    
    fig.update_layout(
        title="Publication Year vs Rating",
        xaxis_title="Year Published",
        yaxis_title="Bayes Average Rating",
        plot_bgcolor=CHART_BG,
        paper_bgcolor=CHART_BG,
        height=400,
        font_color=MUTED
    )
    return fig

def create_bubble_chart(df_filtered: pd.DataFrame, highlight_neighbors=None):
    """Create year vs rating bubble chart with complexity as size"""
    fig = go.Figure()
    
    # Main data points
    fig.add_trace(go.Scatter(
        x=df_filtered["Year Published"],
        y=df_filtered["BayesAvgRating"],
        mode='markers',
        marker=dict(
            size=df_filtered["GameWeight"] * 8 + 5,  # Scale complexity for visibility
            color=MUTED,
            opacity=0.5,
            line=dict(width=1, color='white')
        ),
        text=df_filtered["Name"],
        customdata=df_filtered["GameWeight"],
        hovertemplate="<b>%{text}</b><br>Year: %{x}<br>Rating: %{y:.2f}<br>Complexity: %{customdata:.1f}<extra></extra>",
        name="All Games",
        showlegend=True
    ))
    
    # Highlight neighbors if provided
    if highlight_neighbors is not None and len(highlight_neighbors) > 0:
        fig.add_trace(go.Scatter(
            x=highlight_neighbors["Year Published"],
            y=highlight_neighbors["BayesAvgRating"],
            mode='markers',
            marker=dict(
                size=highlight_neighbors["GameWeight"] * 8 + 10,
                color=ACCENT,
                opacity=0.9,
                line=dict(width=2, color='white')
            ),
            text=highlight_neighbors["Name"],
            customdata=highlight_neighbors["GameWeight"],
            hovertemplate="<b>%{text}</b><br>Year: %{x}<br>Rating: %{y:.2f}<br>Complexity: %{customdata:.1f}<extra></extra>",
            name="Similar Games",
            showlegend=True
        ))
    
    fig.update_layout(
        title="Year vs Rating (Bubble Size = Complexity)",
        xaxis_title="Year Published",
        yaxis_title="Bayes Average Rating",
        plot_bgcolor=CHART_BG,
        paper_bgcolor=CHART_BG,
        height=500,
        font_color=MUTED
    )
    return fig

# ---------------------------------
# Enhanced profile presets
# ---------------------------------
PROFILE_PRESETS = {
    "Family Party": {
        "cats": ["Cat:Family", "Cat:Party"],
        "mechs_on": ["Set Collection", "Hand Management", "Voting", "Take That"],
        "year": CURRENT_YEAR, "min_players": 3, "max_players": 8, "play_time": 30, "min_age": 8, "weight": 1.8
    },
    "Cooperative Strategy": {
        "cats": ["Cat:Thematic", "Cat:Strategy"],
        "mechs_on": ["Cooperative Game", "Action Points", "Variable Player Powers"],
        "year": CURRENT_YEAR, "min_players": 1, "max_players": 4, "play_time": 60, "min_age": 12, "weight": 2.6
    },
    "Heavy Euro": {
        "cats": ["Cat:Strategy", "Economic"],
        "mechs_on": ["Network and Route Building", "Market", "Tech Trees / Tech Tracks", "Worker Placement"],
        "year": CURRENT_YEAR, "min_players": 2, "max_players": 4, "play_time": 120, "min_age": 14, "weight": 3.6
    },
    "Thematic Adventure": {
        "cats": ["Cat:Thematic", "Adventure", "Fantasy"],
        "mechs_on": ["Dice Rolling", "Tile Placement", "Variable Player Powers"],
        "year": CURRENT_YEAR, "min_players": 1, "max_players": 4, "play_time": 90, "min_age": 12, "weight": 2.7
    },
    "Abstract Strategy": {
        "cats": ["Cat:Abstract"],
        "mechs_on": ["Grid Movement", "Pattern Building", "Area Majority / Influence"],
        "year": CURRENT_YEAR, "min_players": 2, "max_players": 2, "play_time": 20, "min_age": 10, "weight": 2.0
    },
    "Deckbuilder": {
        "cats": ["Cat:CGS"],
        "mechs_on": ["Deck Construction", "Hand Management", "Set Collection"],
        "year": CURRENT_YEAR, "min_players": 2, "max_players": 4, "play_time": 45, "min_age": 10, "weight": 2.3
    },
    "Social Deduction": {
        "cats": ["Cat:Party"],
        "mechs_on": ["Hidden Roles", "Voting", "Player Elimination"],
        "year": CURRENT_YEAR, "min_players": 5, "max_players": 10, "play_time": 20, "min_age": 10, "weight": 1.7
    },
}

def median_if_available(df: pd.DataFrame, col: str, mask=None, default=None):
    if col not in df.columns: return default
    data = df.loc[mask, col] if mask is not None else df[col]
    data = pd.to_numeric(data, errors="coerce").dropna()
    return float(data.median()) if len(data) else default

def suggest_from_preset(df: pd.DataFrame, preset_key: str) -> Dict:
    preset = PROFILE_PRESETS.get(preset_key, {})
    mask = pd.Series(True, index=df.index)
    for cat in preset.get("cats", []):
        if cat in df.columns:
            mask &= (df[cat] == 1) | (df[cat] == True)
    
    return {
        "Year Published": preset.get("year", CURRENT_YEAR),
        "Min Players": int(round(median_if_available(df, "Min Players", mask, preset.get("min_players", 2)) or 2)),
        "Max Players": int(round(median_if_available(df, "Max Players", mask, preset.get("max_players", 4)) or 4)),
        "Play Time": int(round(median_if_available(df, "Play Time", mask, preset.get("play_time", 60)) or 60)),
        "Min Age": int(round(median_if_available(df, "Min Age", mask, preset.get("min_age", 10)) or 10)),
        "GameWeight": float(median_if_available(df, "GameWeight", mask, preset.get("weight", 2.5)) or 2.5),
        "mechs_on": preset.get("mechs_on", []),
        "cats": preset.get("cats", []),
    }

# ---------------------------------
# Sidebar setup
# ---------------------------------
st.sidebar.title("🎲 Game Data Controls")
uploaded = st.sidebar.file_uploader("Upload dataset (CSV or Parquet)", type=["csv", "parquet"])
df = load_df(uploaded)

# Sidebar filters
st.sidebar.markdown("---")
year_col = "Year Published"
min_year, max_year = int(df[year_col].min()), int(df[year_col].max())
weight_col = "GameWeight"
min_w, max_w = float(df[weight_col].min()), float(df[weight_col].max())
ptime_col = "Play Time"
min_t, max_t = int(df[ptime_col].min()), int(df[ptime_col].max())

k = st.sidebar.slider("Number of clusters", 2, 15, 8, step=1)
topn = st.sidebar.slider("Nearest neighbors to show", 3, 30, 10, step=1)

st.sidebar.subheader("Data Filters")
yr_rng = st.sidebar.slider("Year range", min_year, max_year, (max(min_year, 2010), max_year))
wt_rng = st.sidebar.slider("Complexity range", float(math.floor(min_w)), float(math.ceil(max_w)), (max(1.0, min_w), min(5.0, max_w)))
pt_rng = st.sidebar.slider("Play time range (min)", min_t, max_t, (min_t, max_t))

# ---------------------------------
# Prepare clustering data
# ---------------------------------
X_all, meta = split_features(df)
scaler, kmeans, pca, labels, coords = fit_clusterer(X_all, k=k)

view = df.copy()
view["Cluster"] = labels
view["PCA1"] = coords[:, 0]
view["PCA2"] = coords[:, 1]

# Apply filters
mask = pd.Series(True, index=view.index)
mask &= view[year_col].between(yr_rng[0], yr_rng[1])
mask &= view[weight_col].between(wt_rng[0], wt_rng[1])
mask &= view[ptime_col].between(pt_rng[0], pt_rng[1])
view_f = view[mask].copy()

# ---------------------------------
# Header
# ---------------------------------
st.title("🎲 Board Game Developer Console")
st.markdown("**Analyze market trends, test game concepts, and discover opportunities in the board game space**")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown('<div class="metric-card"><h4>📊 Games in View</h4><h2>' + f"{len(view_f):,}" + '</h2></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card"><h4>🎯 Clusters</h4><h2>' + f"{len(np.unique(labels))}" + '</h2></div>', unsafe_allow_html=True)
with col3:
    median_rating = float(view_f['BayesAvgRating'].median())
    st.markdown('<div class="metric-card"><h4>⭐ Median Rating</h4><h2>' + f"{median_rating:.2f}" + '</h2></div>', unsafe_allow_html=True)
with col4:
    median_owners = int(view_f['Owned Users'].median())
    st.markdown('<div class="metric-card"><h4>👥 Median Owners</h4><h2>' + f"{median_owners:,}" + '</h2></div>', unsafe_allow_html=True)
with col5:
    median_complexity = float(view_f['GameWeight'].median())
    st.markdown('<div class="metric-card"><h4>🧩 Median Complexity</h4><h2>' + f"{median_complexity:.1f}" + '</h2></div>', unsafe_allow_html=True)

# ---------------------------------
# Main tabs
# ---------------------------------
tab_dashboard, tab_wizard, tab_map, tab_explore = st.tabs([
    "📈 Analytics Dashboard", 
    "🧙‍♂️ Profile Wizard", 
    "🗺️ Cluster Map", 
    "🔭 Cluster Explorer"
])

# ---------------------------------
# Analytics Dashboard Tab
# ---------------------------------
with tab_dashboard:
    st.subheader("📈 Market Analytics Dashboard")
    st.markdown("Explore key relationships in the board game market")
    
    # Create the three main charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = create_complexity_vs_rating_chart(view_f)
        st.plotly_chart(fig1, use_container_width=True)
        
    with col2:
        fig2 = create_year_vs_rating_chart(view_f)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Bubble chart spanning full width
    st.markdown("### Combined Analysis: Year vs Rating vs Complexity")
    fig3 = create_bubble_chart(view_f)
    st.plotly_chart(fig3, use_container_width=True)
    
    # Market insights
    st.markdown("<div class='earthcard'>", unsafe_allow_html=True)
    st.markdown("### 📊 Market Insights")
    
    # Calculate some insights
    recent_games = view_f[view_f["Year Published"] >= 2020]
    old_games = view_f[view_f["Year Published"] < 2010]
    
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        st.markdown("**Quality Trends:**")
        if len(recent_games) > 0 and len(old_games) > 0:
            recent_avg = recent_games["BayesAvgRating"].mean()
            old_avg = old_games["BayesAvgRating"].mean()
            trend = "📈 Rising" if recent_avg > old_avg else "📉 Declining"
            st.write(f"• Rating trend (2020+ vs pre-2010): {trend}")
            st.write(f"• Recent average: {recent_avg:.2f}")
            st.write(f"• Historical average: {old_avg:.2f}")
    
    with insights_col2:
        st.markdown("**Complexity Trends:**")
        if len(recent_games) > 0:
            recent_complexity = recent_games["GameWeight"].mean()
            overall_complexity = view_f["GameWeight"].mean()
            st.write(f"• Recent games complexity: {recent_complexity:.2f}")
            st.write(f"• Overall average: {overall_complexity:.2f}")
            if recent_complexity > overall_complexity:
                st.write("• Trend: Games getting more complex 🧩")
            else:
                st.write("• Trend: Games getting simpler 🎯")
    
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------
# Enhanced Profile Wizard Tab
# ---------------------------------
with tab_wizard:
    st.subheader("🧙‍♂️ Game Profile Wizard")
    st.markdown("Design your game concept and get AI-powered predictions and market analysis")

    # Profile selection
    left_wiz, right_wiz = st.columns([1.2, 1])
    with left_wiz:
        preset_key = st.selectbox("Choose a game archetype", list(PROFILE_PRESETS.keys()), index=1)
        auto = suggest_from_preset(df, preset_key)

    with right_wiz:
        st.markdown("**🎯 Preset Details**")
        st.write("**Categories:** " + (", ".join(auto["cats"]) if auto["cats"] else "None specified"))
        st.write("**Key Mechanics:** " + (", ".join(auto["mechs_on"]) if auto["mechs_on"] else "None specified"))

    # Game details form
    st.markdown("### 🎮 Game Details")
    st.markdown("*Fill in what you know - we'll estimate the rest based on similar games*")
    
    with st.form("enhanced_concept_form"):
        # Basic stats
        c1, c2, c3, c4 = st.columns(4)
        year_published = c1.number_input("Year Published", value=int(auto["Year Published"]), step=1)
        min_players = c2.number_input("Min Players", value=int(auto["Min Players"]), step=1)
        max_players = c3.number_input("Max Players", value=int(auto["Max Players"]), step=1)
        play_time = c4.number_input("Play Time (min)", value=int(auto["Play Time"]), step=5)

        c5, c6, c7, c8 = st.columns(4)
        min_age = c5.number_input("Min Age", value=int(auto["Min Age"]), step=1)
        weight = c6.number_input("Complexity (1-5)", value=float(auto["GameWeight"]), step=0.1, format="%.1f")
        kickstarted = c7.selectbox("Kickstarted?", ["No", "Yes"], index=0)
        best_players = c8.number_input("Best Player Count", value=0, step=1, help="Leave 0 if unknown")

        # Mechanics and themes
        X_cols = list(X_all.columns)
        mech_cols = sorted([c for c in X_cols if c.startswith("Mechanic_") or c in [
            "Deck Construction", "Hand Management", "Worker Placement", "Cooperative Game",
            "Dice Rolling", "Set Collection", "Action Points", "Variable Player Powers"
        ]][:50])  # Limit for UI
        
        theme_cols = sorted([c for c in X_cols if c.startswith("Cat:") or c in [
            "Fantasy", "Adventure", "Economic", "Science Fiction", "War", "Horror"
        ]][:30])  # Limit for UI

        st.markdown("### 🔧 Mechanics & Themes")
        m1, m2 = st.columns(2)
        
        selected_mechs = m1.multiselect(
            "🔧 Select Key Mechanics", 
            mech_cols, 
            default=[m for m in auto["mechs_on"] if m in mech_cols],
            help="Choose the main mechanics that define your game"
        )
        
        selected_themes = m2.multiselect(
            "🎨 Select Themes/Categories", 
            theme_cols, 
            default=[t for t in auto["cats"] if t in theme_cols],
            help="Choose themes and categories that best describe your game"
        )

        submitted = st.form_submit_button("🚀 Analyze Game Concept", type="primary")

    # Analysis results
    if submitted:
        # Build profile
        profile = {
            "Year Published": year_published or CURRENT_YEAR,
            "Min Players": min_players or auto["Min Players"],
            "Max Players": max_players or auto["Max Players"],
            "Play Time": play_time or auto["Play Time"],
            "Min Age": min_age or auto["Min Age"],
            "GameWeight": weight or auto["GameWeight"],
            "Kickstarted": 1 if kickstarted == "Yes" else 0,
            "BestPlayers": best_players or 0,
        }

        # Add mechanics and themes
        for m in set(selected_mechs + auto["mechs_on"]):
            if m in X_cols:
                profile[m] = 1
        for t in set(selected_themes + auto["cats"]):
            if t in X_cols:
                profile[t] = 1

        # Create input vector
        x_input = build_input_vector(X_cols, profile)
        x_scaled = scaler.transform(x_input)
        assigned_cluster = int(kmeans.predict(x_scaled)[0])

        # Get neighbors
        neighbors = nearest_neighbors_in_cluster(
            x_scaled, assigned_cluster, view, X_all, scaler, kmeans, topn=topn
        )

        # Create enhanced neighbor analysis
        st.markdown("---")
        st.markdown("## 🎯 Concept Analysis Results")
        
        # Prediction section
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        st.markdown("### 🔮 AI Predictions")
        
        models = load_models(MODEL_PATHS)
        if models:
            preds = predict_with_models(models, x_input)
            
            pred_col1, pred_col2 = st.columns(2)
            with pred_col1:
                st.markdown("**📊 Rating Predictions**")
                rrf = preds.get("rating_rf")
                rxg = preds.get("rating_xgb")
                if rrf is not None:
                    st.metric("Random Forest Rating", f"{rrf:.2f}", help="Predicted BGG rating")
                if rxg is not None:
                    st.metric("XGBoost Rating", f"{rxg:.2f}", help="Alternative rating prediction")
                    
            with pred_col2:
                st.markdown("**👥 Sales Predictions**")
                srf = preds.get("sales_rf")
                sxg = preds.get("sales_xgb")
                if srf is not None:
                    st.metric("RF Ownership", f"{int(round(srf)):,}", help="Predicted number of owners")
                if sxg is not None:
                    st.metric("XGB Ownership", f"{int(round(sxg)):,}", help="Alternative ownership prediction")
        else:
            st.info("🔧 No prediction models found. Add trained models to ./models/ for AI predictions.")
        
        st.markdown('</div>', unsafe_allow_html=True)

        # Market positioning analysis
        st.markdown("<div class='earthcard'>", unsafe_allow_html=True)
        st.markdown("### 🎯 Market Positioning")
        
        cluster_slice = view[view["Cluster"] == assigned_cluster]
        
        # Compare to cluster medians
        def med(col): 
            return float(pd.to_numeric(cluster_slice[col], errors="coerce").median()) if col in cluster_slice.columns else np.nan

        positioning_insights = []
        
        # Complexity analysis
        concept_weight = float(x_input.iloc[0].get("GameWeight", np.nan))
        cluster_weight = med("GameWeight")
        if np.isfinite(concept_weight) and np.isfinite(cluster_weight):
            if concept_weight > cluster_weight + 0.4:
                positioning_insights.append("🧩 **Higher complexity** than similar games → Appeals to hardcore gamers but smaller market")
            elif concept_weight < cluster_weight - 0.4:
                positioning_insights.append("🎯 **Lower complexity** than similar games → More accessible, broader appeal")
            else:
                positioning_insights.append("⚖️ **Complexity aligns** with similar games → Good market fit")

        # Play time analysis
        concept_time = float(x_input.iloc[0].get("Play Time", np.nan))
        cluster_time = med("Play Time")
        if np.isfinite(concept_time) and np.isfinite(cluster_time):
            if concept_time > cluster_time + 30:
                positioning_insights.append("⏰ **Longer play time** → Higher commitment but potentially deeper experience")
            elif concept_time < cluster_time - 30:
                positioning_insights.append("⚡ **Shorter play time** → Easier to get to table, broader appeal")
            else:
                positioning_insights.append("🕐 **Play time matches** market expectations → Good positioning")

        # Mechanics analysis
        active_mechs = [m for m in profile.keys() if m in X_cols and profile.get(m) == 1 and m.startswith(("Mechanic_", "Deck", "Hand", "Worker", "Cooperative"))]
        if active_mechs:
            mech_str = ", ".join(active_mechs[:5])
            positioning_insights.append(f"🔧 **Key mechanics**: {mech_str}")

        st.write(f"**Assigned to Cluster {assigned_cluster}** with {len(cluster_slice)} similar games")
        for insight in positioning_insights:
            st.write(insight)
        
        st.markdown("</div>", unsafe_allow_html=True)

        # Neighbor analysis with charts
        st.markdown("### 🔍 Similar Games Analysis")
        st.markdown(f"Found **{len(neighbors)}** games most similar to your concept:")

        # Enhanced neighbor table
        neighbor_rows = []
        for _, r in neighbors.iterrows():
            name = r.get("Name", "Unknown")
            year = int(r.get("Year Published", 0))
            rating = r.get("BayesAvgRating", np.nan)
            owners = r.get("Owned Users", np.nan)
            complexity = r.get("GameWeight", np.nan)
            
            # Calculate percentiles for context
            same_year_games = view_f[view_f["Year Published"] == year] if year > 0 else pd.DataFrame()
            rating_pct = year_percentile(same_year_games.get("BayesAvgRating", pd.Series(dtype=float)), rating) if len(same_year_games) > 0 else np.nan
            owners_pct = year_percentile(same_year_games.get("Owned Users", pd.Series(dtype=float)), owners) if len(same_year_games) > 0 else np.nan
            
            neighbor_rows.append({
                "Game": name,
                "Year": year if year > 0 else "Unknown",
                "Rating": f"{rating:.2f}" if not pd.isna(rating) else "N/A",
                "Rating %ile": f"{rating_pct:.0f}%" if not pd.isna(rating_pct) else "N/A",
                "Owners": f"{int(owners):,}" if not pd.isna(owners) else "N/A", 
                "Owners %ile": f"{owners_pct:.0f}%" if not pd.isna(owners_pct) else "N/A",
                "Complexity": f"{complexity:.1f}" if not pd.isna(complexity) else "N/A",
                "Distance": f"{r['__dist']:.3f}"
            })

        neighbors_df = pd.DataFrame(neighbor_rows)
        st.dataframe(neighbors_df, use_container_width=True, hide_index=True)

        # Neighbor-focused charts
        st.markdown("### 📊 Performance Analysis: Your Concept vs Similar Games")
        st.markdown("*Charts below show only games similar to your concept*")
        
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            fig1_neighbors = create_complexity_vs_rating_chart(neighbors, neighbors)
            st.plotly_chart(fig1_neighbors, use_container_width=True)
            
        with chart_col2:
            fig2_neighbors = create_year_vs_rating_chart(neighbors, neighbors)
            st.plotly_chart(fig2_neighbors, use_container_width=True)
        
        # Full-width bubble chart for neighbors
        fig3_neighbors = create_bubble_chart(neighbors, neighbors)
        st.plotly_chart(fig3_neighbors, use_container_width=True)

        # Success patterns analysis
        st.markdown("<div class='earthcard'>", unsafe_allow_html=True)
        st.markdown("### 🏆 Success Patterns in Similar Games")
        
        if len(neighbors) > 0:
            high_rated = neighbors[neighbors["BayesAvgRating"] >= 7.5]
            high_owned = neighbors[neighbors["Owned Users"] >= neighbors["Owned Users"].median()]
            
            success_col1, success_col2 = st.columns(2)
            
            with success_col1:
                st.markdown("**🌟 Highly Rated Games (7.5+)**")
                if len(high_rated) > 0:
                    st.write(f"• {len(high_rated)} of {len(neighbors)} similar games are highly rated")
                    avg_complexity = high_rated["GameWeight"].mean()
                    avg_time = high_rated["Play Time"].mean()
                    st.write(f"• Average complexity: {avg_complexity:.1f}")
                    st.write(f"• Average play time: {avg_time:.0f} minutes")
                else:
                    st.write("• No highly rated games in this cluster")
            
            with success_col2:
                st.markdown("**👥 Popular Games (High Ownership)**")
                if len(high_owned) > 0:
                    st.write(f"• {len(high_owned)} of {len(neighbors)} have above-median ownership")
                    median_owners = int(high_owned["Owned Users"].median())
                    recent_popular = high_owned[high_owned["Year Published"] >= 2018]
                    st.write(f"• Median owners: {median_owners:,}")
                    st.write(f"• {len(recent_popular)} popular games from 2018+")
                else:
                    st.write("• Limited ownership data available")
        
        st.markdown("</div>", unsafe_allow_html=True)

        # Download options
        st.markdown("### 📥 Export Analysis")
        download_col1, download_col2 = st.columns(2)
        
        with download_col1:
            st.download_button(
                "📊 Download Similar Games CSV",
                data=neighbors_df.to_csv(index=False).encode("utf-8"),
                file_name=f"similar_games_{preset_key.lower().replace(' ', '_')}.csv",
                mime="text/csv"
            )
        
        with download_col2:
            # Create summary report
            summary_data = {
                "Concept": [preset_key],
                "Cluster": [assigned_cluster],
                "Predicted_Rating": [preds.get("rating_rf", "N/A") if models else "N/A"],
                "Predicted_Owners": [preds.get("sales_rf", "N/A") if models else "N/A"],
                "Similar_Games_Count": [len(neighbors)],
                "Complexity": [concept_weight],
                "Play_Time": [concept_time]
            }
            summary_df = pd.DataFrame(summary_data)
            st.download_button(
                "📋 Download Analysis Summary",
                data=summary_df.to_csv(index=False).encode("utf-8"),
                file_name=f"concept_analysis_{preset_key.lower().replace(' ', '_')}.csv",
                mime="text/csv"
            )

# ---------------------------------
# Cluster Map Tab (Enhanced)
# ---------------------------------
with tab_map:
    st.subheader("🗺️ Game Landscape Map")
    st.markdown("Explore the board game design space through clustering and PCA visualization")
    
    # Map controls
    map_col1, map_col2 = st.columns([3, 1])
    
    with map_col2:
        color_by_opts = ["Cluster", "Year Published", "GameWeight", "BayesAvgRating"]
        color_by = st.selectbox("Color points by:", color_by_opts, index=0)
        
        show_contours = st.checkbox("Show density contours", value=True)
        point_size = st.slider("Point size", 4, 12, 8)
    
    with map_col1:
        hover_cols = ["Name", "Year Published", "BayesAvgRating", "Owned Users", "GameWeight", "Play Time"]
        hover_data = {col: True for col in hover_cols if col in view_f.columns}

        fig_map = px.scatter(
            view_f, x="PCA1", y="PCA2", color=color_by,
            color_discrete_sequence=PALETTE if color_by == "Cluster" else None,
            hover_data=hover_data,
            height=600,
            title=f"Game Design Space - Colored by {color_by}"
        )
        fig_map.update_traces(marker=dict(size=point_size, opacity=0.8))
        fig_map.update_layout(
            plot_bgcolor=CHART_BG,
            paper_bgcolor=CHART_BG,
            font_color=MUTED
        )

        # Add density contours if requested
        if show_contours and len(view_f) > 50:
            fig_contour = px.density_contour(view_f, x="PCA1", y="PCA2")
            for trace in fig_contour.data:
                trace.update(opacity=0.3, line=dict(color=MUTED))
                fig_map.add_trace(trace)

        st.plotly_chart(fig_map, use_container_width=True)

    # Cluster statistics
    st.markdown("### 📈 Cluster Performance Overview")
    
    cluster_stats = []
    for cluster_id in sorted(view_f["Cluster"].unique()):
        cluster_data = view_f[view_f["Cluster"] == cluster_id]
        
        cluster_stats.append({
            "Cluster": cluster_id,
            "Games": len(cluster_data),
            "Avg Rating": f"{cluster_data['BayesAvgRating'].mean():.2f}",
            "Avg Owners": f"{int(cluster_data['Owned Users'].mean()):,}",
            "Avg Complexity": f"{cluster_data['GameWeight'].mean():.1f}",
            "Avg Play Time": f"{int(cluster_data['Play Time'].mean())} min",
            "Year Range": f"{int(cluster_data['Year Published'].min())}-{int(cluster_data['Year Published'].max())}"
        })
    
    cluster_df = pd.DataFrame(cluster_stats)
    st.dataframe(cluster_df, use_container_width=True, hide_index=True)

# ---------------------------------
# Cluster Explorer Tab (Enhanced)
# ---------------------------------
with tab_explore:
    st.subheader("🔭 Deep Dive: Cluster Analysis")
    
    explore_col1, explore_col2 = st.columns([1, 2])
    
    with explore_col1:
        cluster_pick = st.selectbox("Select cluster to explore:", sorted(view_f["Cluster"].unique()))
        
        cluster_data = view_f[view_f["Cluster"] == cluster_pick].copy()
        
        st.markdown(f"### Cluster {cluster_pick} Overview")
        st.metric("Total Games", len(cluster_data))
        st.metric("Avg Rating", f"{cluster_data['BayesAvgRating'].mean():.2f}")
        st.metric("Avg Complexity", f"{cluster_data['GameWeight'].mean():.1f}")
        st.metric("Avg Play Time", f"{int(cluster_data['Play Time'].mean())} min")
        
        # Top games in cluster
        st.markdown("**🏆 Top Rated Games:**")
        top_games = cluster_data.nlargest(5, "BayesAvgRating")[["Name", "BayesAvgRating", "Year Published"]]
        for _, game in top_games.iterrows():
            st.write(f"• {game['Name']} ({game['Year Published']:.0f}) - {game['BayesAvgRating']:.2f}")
    
    with explore_col2:
        # Cluster visualization
        fig_cluster = px.scatter(
            cluster_data, x="PCA1", y="PCA2",
            size="Owned Users", color="BayesAvgRating",
            hover_data=["Name", "Year Published", "GameWeight"],
            color_continuous_scale="Viridis",
            title=f"Cluster {cluster_pick} - Size by Owners, Color by Rating",
            height=500
        )
        fig_cluster.update_layout(
            plot_bgcolor=CHART_BG,
            paper_bgcolor=CHART_BG,
            font_color=MUTED
        )
        st.plotly_chart(fig_cluster, use_container_width=True)
        
        # Cluster trends over time
        yearly_stats = cluster_data.groupby("Year Published").agg({
            "BayesAvgRating": "mean",
            "Owned Users": "mean", 
            "GameWeight": "mean",
            "Name": "count"
        }).reset_index()
        yearly_stats.rename(columns={"Name": "Game Count"}, inplace=True)
        
        if len(yearly_stats) > 1:
            fig_trends = px.line(
                yearly_stats, x="Year Published", y="BayesAvgRating",
                title=f"Rating Trends in Cluster {cluster_pick}",
                height=300
            )
            fig_trends.update_layout(
                plot_bgcolor=CHART_BG,
                paper_bgcolor=CHART_BG,
                font_color=MUTED
            )
            st.plotly_chart(fig_trends, use_container_width=True)

    # Detailed game list for the cluster
    st.markdown("### 📋 All Games in Cluster")
    display_cols = ["Name", "Year Published", "BayesAvgRating", "Owned Users", "GameWeight", "Play Time"]
    cluster_display = cluster_data[display_cols].sort_values("BayesAvgRating", ascending=False)
    st.dataframe(cluster_display, use_container_width=True, hide_index=True)

# ---------------------------------
# Footer
# ---------------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #6B705C; padding: 1rem;'>
    🎲 <strong>Board Game Developer Console</strong> | 
    Built for data-driven game design decisions | 
    Upload your own dataset or use the default BGG data
    </div>
    """, 
    unsafe_allow_html=True
)
