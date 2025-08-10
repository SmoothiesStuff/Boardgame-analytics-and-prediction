# streamlit_app.py — Advanced Board Game Developer Console with Enhanced Analytics
import os
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "poll")
os.environ.setdefault("WATCHDOG_FORCE_POLLING", "true")
import math
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from streamlit_plotly_events import plotly_events
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from scipy import stats
from joblib import load as joblib_load
import warnings
import random, hashlib, json

RANDOM_SEED = 1337
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
warnings.filterwarnings('ignore')

# Page config & theme
st.set_page_config(
    page_title="Board Game Developer Console - Professional Edition", 
    page_icon="🎲", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced color palette with more professional tones
PALETTE = [
    "#2E8B57", "#CD853F", "#5F9EA0", "#BC8F8F", "#4682B4",
    "#8B4513", "#6B8E23", "#D2691E", "#708090", "#F4A460",
    "#8B7355", "#A67C52", "#7C8C4A", "#C4A484", "#4F4538",
    "#B56576", "#6B705C", "#D4A373", "#8D6B94", "#9C8B7A"
]

CHART_COLORS = [
    "#2E8B57", "#CD853F", "#5F9EA0", "#BC8F8F", "#4682B4",
    "#8B4513", "#6B8E23", "#D2691E", "#708090", "#F4A460"
]

ACCENT = "#2E8B57"
SECONDARY = "#CD853F"
MUTED = "#6B705C"
BG_SOFT = "#FAF7F2"
CHART_BG = "#FFFCF7"
SUCCESS_COLOR = "#2E8B57"
WARNING_COLOR = "#D2691E"
DANGER_COLOR = "#BC8F8F"

st.markdown(f"""
<style>
.stApp {{ background-color: {BG_SOFT}; }}
section[data-testid="stSidebar"] {{ background-color: #F3EFE7; }}
h1, h2, h3, h4 {{ color: {ACCENT}; font-family: 'Georgia', serif; }}
.main-header {{ 
    background: linear-gradient(135deg, {ACCENT} 0%, {SECONDARY} 100%);
    color: white;
    padding: 2rem;
    border-radius: 12px;
    margin-bottom: 2rem;
}}
.earthcard {{ 
    border: 1px solid #e5dcc9; 
    border-radius: 12px; 
    padding: 1.5rem; 
    background: {CHART_BG}; 
    margin: 1rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}}
.metric-card {{
    background: {CHART_BG};
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid {ACCENT};
    margin: 0.5rem 0;
    transition: transform 0.2s;
}}
.metric-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}}
.prediction-card {{
    background: linear-gradient(135deg, {CHART_BG} 0%, #F5F1EA 100%);
    padding: 1.5rem;
    border-radius: 12px;
    border: 2px solid {ACCENT};
    margin: 1rem 0;
}}
.insight-box {{
    background: linear-gradient(90deg, {CHART_BG} 0%, #FFF 100%);
    border-left: 5px solid {ACCENT};
    padding: 1rem 1.5rem;
    margin: 1rem 0;
    border-radius: 0 8px 8px 0;
}}
.recommendation-card {{
    background: {CHART_BG};
    border: 2px dashed {SECONDARY};
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
}}
.stTabs [data-baseweb="tab-list"] {{
    gap: 24px;
}}
.stTabs [data-baseweb="tab"] {{
    padding: 12px 24px;
    font-weight: 600;
}}
</style>
""", unsafe_allow_html=True)
st.markdown(f"""
<style>
.stApp blockquote {{
    background: linear-gradient(90deg, {CHART_BG} 0%, #FFF 100%);
    border-left: 5px solid {ACCENT};
    padding: 1rem 1.5rem;
    margin: 1rem 0;
    border-radius: 0 8px 8px 0;
    color: {MUTED};
}}
</style>
""", unsafe_allow_html=True)

# Configuration
DEFAULT_PARQUET_PATH = "cleaned_large_bgg_dataset.parquet"
DEFAULT_CSV_PATH = "cleaned_large_bgg_dataset.csv"
CURRENT_YEAR = 2021

EXCLUDE_FOR_CLUSTERING = [
    "Owned Users", "BayesAvgRating", "AvgRating", "Users Rated", "BGG Rank", 
    "StdDev", "NumWant", "NumWish", "NumComments", "NumWeightVotes",
    "ID", "BGGId", "Name", "ImagePath", "Rank:strategygames", "Rank:abstracts", 
    "Rank:familygames", "Rank:thematic", "Rank:cgs", "Rank:wargames", 
    "Rank:partygames", "Rank:childrensgames"
]

MODEL_PATHS = {
    # "rating_xgb": "models/rating_xgb.joblib", #old models, not so gooood
    # "sales_xgb": "models/sales_xgb.joblib",
    "rating_xgb": "models/train_model_rating_xgb.joblib",
    "owned_rf":   "models/train_model_owned_rf.joblib",          # or leave missing if you used XGB Poisson
    "owned_xgb":  "models/train_model_owned_xgb_poisson.joblib", # optional fallback 
}
INPUT_SCALER_PATH = "models/input_scaler.joblib"  
# Features not used for training
PRED_EXCLUDE = {
    "ID","Name","BGGId","Users Rated","Rating Average","BayesAvgRating","AvgRating",
    "Owned Users","LogOwned","BGG Rank","StdDev",
    "NumWish","NumWant","NumComments","NumWeightVotes",
    "Rank:strategygames","Rank:abstracts","Rank:familygames","Rank:thematic",
    "Rank:cgs","Rank:wargames","Rank:partygames","Rank:childrensgames",
    "Cluster","PCA1","PCA2","SalesPercentile","__dist"
}

# Enhanced data loading with caching
@st.cache_data(show_spinner=True, ttl=3600)
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

# Enhanced input alignment with better type handling
ONEHOT_PREFIXES = ["Cat:", "Mechanic_", "Fantasy", "Adventure", "Economic", "Science Fiction", "War", "Horror"]
NUMERIC_COLS = [
    "Year Published","Min Players","Max Players","Play Time","Min Age",
    "GameWeight","Kickstarted","BestPlayers","Rating Average","Complexity Average"
]
# --- Theme discovery (only use columns present in X_all, so models align) ---
THEME_CANDIDATES = [
    # Core BGG “super-cats”
    "Cat:Strategy", "Cat:Thematic", "Cat:Family", "Cat:Party", "Cat:Abstract", "Cat:Wargame",
    # Broad/popular themes
    "Cat:Fantasy", "Cat:Science Fiction", "Cat:Adventure", "Cat:Economic", "Cat:Horror", "Cat:War",
    "Cat:City Building", "Cat:Civilization", "Cat:Animals", "Cat:Mythology", "Cat:Historical",
    "Cat:Nautical", "Cat:Pirates", "Cat:Space Exploration", "Cat:Post-Apocalyptic", "Cat:Steampunk",
    "Cat:Western", "Cat:Zombies", "Cat:Mystery", "Cat:Crime", "Cat:Humor",
    # Domain/interest clusters
    "Cat:Transportation", "Cat:Trains", "Cat:Aviation", "Cat:Automobile", "Cat:Sports", "Cat:Medical",
    "Cat:Industry / Manufacturing", "Cat:Educational", "Cat:Environmental", "Cat:Travel",
    # Format-ish categories you sometimes see in datasets
    "Cat:Card Game", "Cat:Dice", "Cat:Miniatures", "Cat:Territory Building", "Cat:Negotiation",
    # Non-prefixed fallbacks some datasets include
    "Fantasy", "Adventure", "Economic", "Science Fiction", "War", "Horror"
]

def discover_available_themes(X_cols, preset_cats=None, limit=60):
    cols = set(X_cols)
    # Only keep candidates that are real columns
    base = [c for c in THEME_CANDIDATES if c in cols]
    # Include any extra Cat: columns in your data we didn’t list
    extras = sorted([c for c in cols if isinstance(c, str) and c.startswith("Cat:") and c not in base])
    # Nudge preset categories to the front if present
    preset = [c for c in (preset_cats or []) if c in cols]
    # De-dup while preserving order: preset → base → extras
    seen, ordered = set(), []
    for group in (preset, base, extras):
        for c in group:
            if c not in seen:
                ordered.append(c); seen.add(c)
    return ordered[:limit]
# ---------- Toggle helpers ----------
def discover_mechanics(cols):
    base = [c for c in cols if isinstance(c, str) and c.startswith("Mechanic_")]
    extras = [c for c in [
        "Cooperative Game","Variable Player Powers","Deck Construction","Hand Management",
        "Worker Placement","Dice Rolling","Set Collection","Action Points","Hidden Roles",
        "Tile Placement","Area Majority / Influence","Pattern Building","Grid Movement",
        "Market","Network and Route Building","Auction/Bidding"
    ] if c in cols]
    seen, ordered = set(), []
    for c in base + extras:
        if c not in seen:
            ordered.append(c); seen.add(c)
    return ordered

def discover_themes(cols):
    base = [c for c in cols if isinstance(c, str) and c.startswith("Cat:")]
    extras = [c for c in [
        "Fantasy","Adventure","Economic","Science Fiction","War","Horror","City Building",
        "Civilization","Animals","Mythology","Historical","Nautical","Pirates",
        "Space Exploration","Post-Apocalyptic","Steampunk","Western","Zombies","Mystery",
        "Crime","Humor","Transportation","Trains","Aviation","Automobile","Sports","Medical",
        "Industry / Manufacturing","Educational","Environmental","Travel","Card Game","Dice",
        "Miniatures","Territory Building","Negotiation"
    ] if c in cols]
    seen, ordered = set(), []
    for c in base + extras:
        if c not in seen:
            ordered.append(c); seen.add(c)
    return ordered

def render_toggle_grid(options, defaults=None, columns=3, key_prefix="tg"):
    """Render options as checkboxes in a grid; returns list of selected option names (exact column names)."""
    defaults = set(defaults or [])
    sels = []
    cols = st.columns(columns)
    for i, opt in enumerate(options):
        with cols[i % columns]:
            label = opt.replace("Mechanic_", "").replace("Cat:", "")
            checked = st.checkbox(label, value=(opt in defaults), key=f"{key_prefix}_{i}_{opt}")
            if checked:
                sels.append(opt)
    return sels

def toggle_feature(profile: dict, X_cols, on: bool, names):
    """Flip a binary feature to 1 only if it exists in X_all."""
    if not on:
        return
    if isinstance(names, str):
        names = [names]
    cols = set(X_cols)
    for name in names:
        if name in cols:
            profile[name] = 1
        else:
            # try Mechanic_ prefix form
            alt = f"Mechanic_{name.replace(' ', '_')}"
            if alt in cols:
                profile[alt] = 1


def _predict_agnostic(model, X_df, feature_names):
# Try sklearn-style first
    try:
        return float(model.predict(X_df.values)[0])
    except Exception:
        pass
    # Try native XGBoost Booster
    try:
        import xgboost as xgb
        dtest = xgb.DMatrix(X_df.values, feature_names=list(feature_names))
        it = getattr(model, "best_iteration", None)
        if it is not None:
            try:
                return float(model.predict(dtest, iteration_range=(0, it + 1))[0])
            except TypeError:
                pass
        ntree = getattr(model, "best_ntree_limit", 0)
        if ntree:
            return float(model.predict(dtest, ntree_limit=ntree)[0])
        return float(model.predict(dtest)[0])
    except Exception:
        raise
        
def align_profile_to_training(profile: dict, training_cols: list[str], scaler=None) -> pd.DataFrame:
    """Enhanced profile alignment with robust type handling and validation."""
    training_cols = [c for c in training_cols if c not in PRED_EXCLUDE]
    
    # Type coercion
    clean = {}
    for k, v in profile.items():
        if isinstance(v, str) and k in NUMERIC_COLS:
            v = v.replace(",", ".")
        if k in NUMERIC_COLS:
            try:
                clean[k] = float(v)
            except:
                clean[k] = 0.0
        else:
            clean[k] = v
    
    # Consistency checks
    if "Min Players" in clean and "Max Players" in clean:
        if clean["Min Players"] > clean["Max Players"]:
            clean["Max Players"] = clean["Min Players"]
    
    # Build row with defaults
    row = {c: 0.0 for c in training_cols}
    row.update({k: clean.get(k, 0.0) for k in training_cols})
    
    X = pd.DataFrame([row], columns=training_cols)
    
    # Ensure at least one feature is on in each one-hot group
    for prefix in ONEHOT_PREFIXES:
        group = [c for c in training_cols if c.startswith(prefix)]
        if group and X[group].sum(axis=1).iloc[0] == 0:
            X.at[X.index[0], group[0]] = 1.0
    
    # Clamp to training range if scaler provided
    if scaler and hasattr(scaler, "data_min_"):
        num_cols = [c for c in training_cols if c in NUMERIC_COLS]
        if num_cols:
            idx = [training_cols.index(c) for c in num_cols]
            arr = X[training_cols].astype(float).values
            lo = scaler.data_min_[idx]
            hi = scaler.data_max_[idx]
            arr[:, idx] = np.clip(arr[:, idx], lo, hi)
            X[training_cols] = arr
    
    return X

@st.cache_resource(show_spinner=False)
def fit_clusterer(X: pd.DataFrame, k: int = 8, random_state: int = 42):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determine effective k
    n_unique = min(len(X), np.unique(X_scaled, axis=0).shape[0])
    k_eff = max(2, min(k, n_unique, len(X)))
    
    if k_eff < 2:
        labels = np.zeros(len(X), dtype=int)
        coords = np.c_[X_scaled[:, :1], np.zeros((len(X_scaled), 1))]
        class _DummyK:
            def predict(self, Z): return np.zeros(len(Z), dtype=int)
        return scaler, _DummyK(), None, labels, coords
    
    kmeans = KMeans(n_clusters=k_eff, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    # PCA for visualization
    n_comp = min(2, X_scaled.shape[1], len(X_scaled))
    if n_comp >= 2:
        pca = PCA(n_components=2, random_state=random_state)
        coords = pca.fit_transform(X_scaled)
    else:
        coords = np.c_[X_scaled[:, :1], np.zeros((len(X_scaled), 1))]
        pca = None
    
    return scaler, kmeans, pca, labels, coords
########## cluster naming utils ##########
def _nice_list(words, max_n=2):                                                # make "A + B" strings
    words = [w for w in words if w]
    return " + ".join(words[:max_n]) if words else ""

def _top_diff_features(cluster_df, all_df, startswith, min_prop=0.25, max_show=2):
    cols = [c for c in cluster_df.columns if c.startswith(startswith)]
    if not cols:
        return []
    # usage gaps vs overall
    diffs = []
    overall = all_df[cols].mean().fillna(0.0)
    local   = cluster_df[cols].mean().fillna(0.0)
    for c in cols:
        if local[c] >= min_prop:
            diffs.append((c, float(local[c] - overall[c])))
    diffs.sort(key=lambda x: x[1], reverse=True)
    names = [c.replace("Mechanic_", "").replace("Cat:", "").replace("_", " ") for c,_ in diffs]
    # small cleanups
    names = [n.replace("Cgs", "Deck/CCG").replace("Cgs ", "Deck/CCG ") for n in names]
    return names[:max_show]

def generate_cluster_labels(view_df: pd.DataFrame) -> dict[int, str]:
    labels = {}
    # global percentiles for context
    w_lo, w_hi = np.nanpercentile(view_df["GameWeight"], [33, 67])
    r_lo, r_hi = 6.3, 7.3
    t_lo, t_hi = np.nanpercentile(view_df["Play Time"], [33, 67])

    for cid, cdf in view_df.groupby("Cluster"):
        if cdf.empty:
            labels[cid] = f"Segment {cid}"
            continue
        # adjectives by position vs percentiles
        w = float(cdf["GameWeight"].median())
        r = float(cdf["AvgRating"].mean())
        t = float(cdf["Play Time"].median())
        comp = "Low-complexity" if w < w_lo else ("High-complexity" if w > w_hi else "Mid-weight")
        qual = "weakly rated" if r < r_lo else ("strongly rated" if r >= r_hi else "solid-rating")
        dur  = "short" if t < t_lo else ("long" if t > t_hi else "medium")

        mechs  = _top_diff_features(cdf, view_df, "Mechanic_", min_prop=0.25, max_show=2)
        themes = _top_diff_features(cdf, view_df, "Cat:",       min_prop=0.25, max_show=1)

        mech_str  = _nice_list(mechs)
        theme_str = _nice_list(themes, max_n=1)

        bits = [comp, qual]
        if theme_str: bits.append(theme_str.lower())
        if mech_str:  bits.append(f"with {mech_str}")
        bits.append(f"({dur} play)")

        name = " ".join(bits).strip()
        # fallback if too bare
        if len(name) < 12:
            top_example = cdf.nlargest(1, "AvgRating")["Name"].iloc[0]
            name = f"Segment {cid}: similar to {top_example}"
        labels[cid] = name[0].upper() + name[1:]
    return labels

# Enhanced analytics functions
def calculate_market_opportunity_score(cluster_df: pd.DataFrame, year: int = CURRENT_YEAR) -> float:
    """Calculate market opportunity score for a cluster based on multiple factors."""
    recent_games = cluster_df[cluster_df["Year Published"] >= year - 3]
    
    # Factor 1: Growth rate
    if len(cluster_df) > 5:
        years = cluster_df.groupby("Year Published").size()
        if len(years) > 2:
            recent_growth = years.iloc[-3:].mean() / max(1, years.iloc[-6:-3].mean())
        else:
            recent_growth = 1.0
    else:
        recent_growth = 1.0
    
    # Factor 2: Average performance
    avg_rating = cluster_df["AvgRating"].mean()
    avg_owners = cluster_df["Owned Users"].median()
    
    # Factor 3: Market saturation (inverse)
    saturation = len(recent_games) / max(1, len(cluster_df))
    
    # Factor 4: Success rate
    success_rate = (cluster_df["AvgRating"] >= 7.0).mean()
    
    # Weighted score
    opportunity = (
        recent_growth * 0.3 +
        (avg_rating / 10) * 0.25 +
        (min(avg_owners / 10000, 1)) * 0.2 +
        (1 - saturation) * 0.15 +
        success_rate * 0.1
    )
    
    return min(max(opportunity * 100, 0), 100)
########## pricing helpers ##########
def estimate_anchor_price(complexity: float, component_quality: str, production_quality: str,
                          max_players: int, play_time_min: int) -> float:
    # Rough MSRP baseline from design signals (tunable)
    base = 35.0 + (complexity - 2.5) * 6.0
    if component_quality == "Premium": base += 15
    elif component_quality == "Good": base += 5
    if production_quality in ["Premium", "Deluxe"]: base += 10
    if max_players >= 5: base += 5
    if play_time_min >= 90: base += 5
    return float(np.clip(base, 15, 150))

def default_channel_fee_pct(funding_model: str) -> float:
    # Typical all-in cuts (rough, adjustable)
    if funding_model == "Traditional": return 0.50   # retail/wholesale stack
    if funding_model in ["Kickstarter", "Gamefound"]: return 0.12  # platform + processing
    return 0.40

def identify_mechanic_synergies(df: pd.DataFrame, min_games: int = 10) -> pd.DataFrame:
    """Identify successful mechanic combinations."""
    mech_cols = [c for c in df.columns if c.startswith("Mechanic_") or c in [
        "Deck Construction", "Hand Management", "Worker Placement", "Cooperative Game"
    ]]
    
    synergies = []
    for i, mech1 in enumerate(mech_cols):
        for mech2 in mech_cols[i+1:]:
            combo_games = df[(df[mech1] == 1) & (df[mech2] == 1)]
            if len(combo_games) >= min_games:
                synergies.append({
                    "Mechanic 1": mech1.replace("Mechanic_", ""),
                    "Mechanic 2": mech2.replace("Mechanic_", ""),
                    "Games": len(combo_games),
                    "Avg Rating": combo_games["AvgRating"].mean(),
                    "Success Rate": (combo_games["AvgRating"] >= 7.5).mean(),
                    "Avg Owners": combo_games["Owned Users"].median()
                })
    
    return pd.DataFrame(synergies).sort_values("Success Rate", ascending=False)

def generate_design_recommendations(cluster_df: pd.DataFrame, overall_df: pd.DataFrame) -> List[str]:
    """Generate specific, actionable design recommendations."""
    recommendations = []
    
    # Complexity sweet spot
    optimal_complexity = cluster_df[cluster_df["AvgRating"] >= 7.5]["GameWeight"].median()
    if not pd.isna(optimal_complexity):
        recommendations.append(f"🎯 **Target complexity: {optimal_complexity:.1f}/5.0** - This is the sweet spot for highly-rated games in this category")
    
    # Play time optimization
    successful_games = cluster_df[cluster_df["AvgRating"] >= 7.0]
    if len(successful_games) > 0:
        optimal_time = successful_games["Play Time"].median()
        recommendations.append(f"⏱️ **Optimal play time: {int(optimal_time)} minutes** - Successful games in this space hit this duration")
    
    # Underused successful mechanics
    mech_cols = [c for c in cluster_df.columns if c.startswith("Mechanic_")]
    for mech in mech_cols:
        usage_rate = cluster_df[mech].mean()
        if 0.1 < usage_rate < 0.3:  # Underused but present
            success_with = cluster_df[cluster_df[mech] == 1]["AvgRating"].mean()
            success_without = cluster_df[cluster_df[mech] == 0]["AvgRating"].mean()
            if success_with > success_without + 0.3:
                mech_name = mech.replace("Mechanic_", "").replace("_", " ")
                recommendations.append(f"💡 **Consider {mech_name}** - Only {usage_rate*100:.0f}% of games use it, but they rate {success_with - success_without:.1f} points higher")
    
    # Market gaps
    recent_years = cluster_df[cluster_df["Year Published"] >= CURRENT_YEAR - 2]
    if len(recent_years) < len(cluster_df) * 0.1:
        recommendations.append(f"🚀 **Market opportunity** - This category has seen only {len(recent_years)} releases in the past 2 years (historical average: {len(cluster_df)/10:.0f}/year)")
    
    return recommendations[:5]  # Top 5 recommendations

# Enhanced visualization functions
def create_market_evolution_timeline(df: pd.DataFrame) -> go.Figure:
    """Create comprehensive market evolution timeline with multiple metrics."""
    yearly = df.groupby("Year Published").agg({
        "Name": "count",
        "AvgRating": "mean",
        "GameWeight": "mean",
        "Owned Users": "median",
        "Play Time": "mean"
    }).reset_index()
    yearly.rename(columns={"Name": "Games Released"}, inplace=True)
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("Games Released per Year", "Average Quality and Complexity", "Ownership Vs Playtime"),
        vertical_spacing=0.12,
        specs=[[{"secondary_y": False}], [{"secondary_y": True}], [{"secondary_y": True}]]
    )
    
    # Row 1: Release volume
    fig.add_trace(
        go.Bar(x=yearly["Year Published"], y=yearly["Games Released"],
               name="Games Released", marker_color=CHART_COLORS[0], opacity=0.7),
        row=1, col=1
    )
    
    # Row 2: Quality metrics
    fig.add_trace(
        go.Scatter(x=yearly["Year Published"], y=yearly["AvgRating"],
                   mode='lines+markers', name='Avg Rating',
                   line=dict(color=CHART_COLORS[1], width=3)),
        row=2, col=1, secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=yearly["Year Published"], y=yearly["GameWeight"],
                   mode='lines+markers', name='Avg Complexity',
                   line=dict(color=CHART_COLORS[2], width=2, dash='dot')),
        row=2, col=1, secondary_y=True
    )
    
    # Row 3: Market metrics
    fig.add_trace(
        go.Scatter(x=yearly["Year Published"], y=yearly["Owned Users"],
                   mode='lines+markers', name='Median Owners',
                   line=dict(color=CHART_COLORS[3], width=3)),
        row=3, col=1, secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=yearly["Year Published"], y=yearly["Play Time"],
                   mode='lines+markers', name='Avg Play Time (min)',
                   line=dict(color=CHART_COLORS[4], width=2, dash='dash')),
        row=3, col=1, secondary_y=True
    )
    
    fig.update_xaxes(title_text="Year", row=3, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Rating", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Complexity", row=2, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Owners", row=3, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Minutes", row=3, col=1, secondary_y=True)
    
    fig.update_layout(
        height=800,
        plot_bgcolor=CHART_BG,
        paper_bgcolor=CHART_BG,
        font_color=MUTED,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

def build_mech_network_fig_static(synergies_df: pd.DataFrame, top_n: int = 30) -> go.Figure:
    if synergies_df is None or len(synergies_df) == 0:
        return go.Figure().add_annotation(text="Insufficient data for network graph", showarrow=False)

    df = synergies_df.copy().sort_values(["Success Rate", "Games"], ascending=False).head(top_n).reset_index(drop=True)

    # Nodes and circular layout
    nodes = sorted(set(df["Mechanic 1"]).union(set(df["Mechanic 2"])))
    n = len(nodes)
    if n == 0:
        return go.Figure().add_annotation(text="No mechanics to display", showarrow=False)

    node_idx = {name: i for i, name in enumerate(nodes)}
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = np.cos(theta)
    y = np.sin(theta)

    # Edge visual scaling
    s = df["Success Rate"].astype(float).clip(0, 1)
    if (s.max() - s.min()) < 0.05 and len(s) > 1:
        m = s.rank(method="dense", pct=True)     # keep visible spread when range is tiny
    else:
        m = (s - s.min()) / max(1e-9, s.max() - s.min())

    widths = 2.0 + m * (10.0 - 2.0)
    edge_colors = [px.colors.sample_colorscale("YlGnBu", float(v))[0] for v in m]

    fig = go.Figure()

    # --- Edges first (under the nodes) ---
    for i, row in df.iterrows():
        a, b = row["Mechanic 1"], row["Mechanic 2"]
        ia, ib = node_idx[a], node_idx[b]
        fig.add_trace(go.Scatter(
            x=[x[ia], x[ib]], y=[y[ia], y[ib]],
            mode="lines",
            line=dict(width=float(widths[i]), color=edge_colors[i]),
            opacity=0.75,
            hovertemplate=f"{a} + {b} • SR {row['Success Rate']*100:.0f}% • n={int(row['Games'])}<extra></extra>",
            showlegend=False
        ))

    # --- Nodes on top (always visible) ---
    label_text = [n.replace("Mechanic_", "") for n in nodes]
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="markers+text",
        text=label_text,
        textposition="top center",
        hovertemplate="%{text}<extra></extra>",
        marker=dict(
            size=22,                             # fixed, visible size
            color=ACCENT,                        # your theme color
            opacity=1.0,
            symbol="circle",
            line=dict(width=2, color="rgba(0,0,0,0.35)")  # outline so they never disappear
        ),
        showlegend=False
    ))

    # Layout / axes
    fig.update_layout(
        title="Mechanic Synergy Network",
        height=540,
        plot_bgcolor=CHART_BG,
        paper_bgcolor=CHART_BG,
        font_color=MUTED,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        margin=dict(t=60, r=20, l=20, b=20),
        hovermode="closest"
    )
    # ---------- Legend (put this after the node trace, before return) ----------
    # 1) Edge thickness legend (use a mid-tone color; widths map to "strength")
    legend_color = px.colors.sample_colorscale("YlGnBu", 0.6)[0]
    for label, w in [("SR ≈ 20%", 3), ("SR ≈ 50%", 6), ("SR ≈ 80%", 10)]:
        fig.add_trace(go.Scatter(
            x=[None, None], y=[None, None],
            mode="lines",
            line=dict(width=w, color=legend_color),
            name=label,
            hoverinfo="skip",
            showlegend=True
        ))

    # 2) Colorbar for Success Rate (0–1); invisible marker just to show the scale
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(
            size=0.0001,                    # effectively invisible
            color=[0, 1],                   # min/max for the scale
            cmin=0, cmax=1,
            colorscale="YlGnBu",
            showscale=True,
            colorbar=dict(title="Success rate", ticksuffix="")
        ),
        hoverinfo="skip",
        showlegend=False
    ))
    
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0)
    )
    return fig


def create_mechanic_network_graph(synergies_df: pd.DataFrame, top_n: int = 30) -> go.Figure:
    """Network graph where edge width & color reflect success rate.
       Visual scaling only; underlying numbers unchanged.
    """
    if synergies_df is None or len(synergies_df) == 0:
        return go.Figure().add_annotation(text="Insufficient data for network graph", showarrow=False)

    # Take the strongest combos first
    df = synergies_df.copy().sort_values(["Success Rate", "Games"], ascending=False).head(top_n).reset_index(drop=True)

    # Nodes
    nodes = sorted(set(df["Mechanic 1"]).union(set(df["Mechanic 2"])))
    node_idx = {n: i for i, n in enumerate(nodes)}

    # Simple circular layout
    n = len(nodes)
    theta = np.linspace(0, 2*np.pi, n, endpoint=False) if n else np.array([])
    x = np.cos(theta); y = np.sin(theta)

    # ---- Edge visual scaling ----
    s = df["Success Rate"].astype(float).clip(0, 1)               # 0–1
    # If the range is narrow, stretch by rank (visual only)
    if (s.max() - s.min()) < 0.05 and len(s) > 1:
        m = s.rank(method="dense", pct=True)                       # 0–1
    else:
        denom = max(1e-9, (s.max() - s.min()))
        m = (s - s.min()) / denom                                  # 0–1

    width_min, width_max = 1.5, 10.0
    widths   = width_min + m * (width_max - width_min)
    opac_min, opac_max = 0.25, 0.85
    opacities = opac_min + m * (opac_max - opac_min)

    # Colors from a continuous scale (stronger = darker)
    edge_colors = [px.colors.sample_colorscale("YlGnBu", float(val))[0] for val in m]

    fig = go.Figure()

    # Edges
    for i, row in df.iterrows():
        a, b = node_idx[row["Mechanic 1"]], node_idx[row["Mechanic 2"]]
        fig.add_trace(go.Scatter(
            x=[x[a], x[b]], y=[y[a], y[b]],
            mode="lines",
            line=dict(width=float(widths[i]), color=edge_colors[i]),
            opacity=float(opacities[i]),
            hovertemplate=(
                f"{row['Mechanic 1']} + {row['Mechanic 2']}"
                f"<br>Success rate: {row['Success Rate']*100:.1f}%"
                f"<br>Games: {int(row['Games'])}<extra></extra>"
            ),
            showlegend=False
        ))

    # Node sizes by total connection strength
    strength = np.zeros(n)
    for _, row in df.iterrows():
        strength[node_idx[row["Mechanic 1"]]] += float(row["Success Rate"])
        strength[node_idx[row["Mechanic 2"]]] += float(row["Success Rate"])

    size_min, size_max = 14.0, 30.0
    if strength.max() > 0:
        node_sizes = size_min + (strength - strength.min()) / (strength.max() - strength.min() + 1e-9) * (size_max - size_min)
    else:
        node_sizes = np.full(n, 18.0)

    # Nodes
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="markers+text",
        marker=dict(size=node_sizes, color=CHART_COLORS[1]),
        text=[n.replace("Mechanic_", "") for n in nodes],
        textposition="top center",
        hovertemplate="%{text}<extra></extra>",
        showlegend=False
    ))

    # Dummy trace to show a colorbar for success rate
    if len(s) > 0:
        cmin, cmax = float(s.min()), float(s.max())
        fig.add_trace(go.Scatter(
            x=[None, None], y=[None, None],
            mode="markers",
            marker=dict(
                colorscale="YlGnBu",
                cmin=cmin, cmax=cmax,
                color=[cmin, cmax],
                size=0,
                colorbar=dict(title="Success rate", tickformat=".0%")
            ),
            hoverinfo="skip",
            showlegend=False
        ))

    fig.update_layout(
        title="Mechanic Synergy Network (edge width & color ~ success rate)",
        height=520,
        plot_bgcolor=CHART_BG,
        paper_bgcolor=CHART_BG,
        font_color=MUTED,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False)
    )
    return fig

def create_success_predictor_chart(df: pd.DataFrame, x_col: str, y_col: str = "AvgRating") -> go.Figure:
    """Create scatter plot with success prediction overlay."""
    fig = go.Figure()
    
    # Calculate success zones
    high_success = df[df[y_col] >= 7.5]
    medium_success = df[(df[y_col] >= 6.5) & (df[y_col] < 7.5)]
    low_success = df[df[y_col] < 6.5]
    
    # Add traces for different success levels
    for data, name, color in [
        (low_success, "Lower Rated (<6.5)", DANGER_COLOR),
        (medium_success, "Medium Rated (6.5-7.5)", WARNING_COLOR),
        (high_success, "Highly Rated (>7.5)", SUCCESS_COLOR)
    ]:
        if len(data) > 0:
            fig.add_trace(go.Scatter(
                x=data[x_col],
                y=data[y_col],
                mode='markers',
                name=name,
                marker=dict(size=6, color=color, opacity=0.6),
                text=data["Name"] if "Name" in data.columns else None,
                hovertemplate=f"<b>%{{text}}</b><br>{x_col}: %{{x:.2f}}<br>Rating: %{{y:.2f}}<extra></extra>"
            ))
    
    # Add trend line
    if len(df) > 5:
        z = np.polyfit(df[x_col].dropna(), df[y_col].dropna(), 2)
        p = np.poly1d(z)
        x_range = np.linspace(df[x_col].min(), df[x_col].max(), 100)
        fig.add_trace(go.Scatter(
            x=x_range,
            y=p(x_range),
            mode='lines',
            name='Success Trend',
            line=dict(color='black', width=2, dash='dash'),
            showlegend=True
        ))
    
    fig.update_layout(
        title=f"Success Zones: {x_col} Impact on Ratings",
        xaxis_title=x_col,
        yaxis_title="Average Rating",
        plot_bgcolor=CHART_BG,
        paper_bgcolor=CHART_BG,
        height=400,
        font_color=MUTED,
        hovermode='closest'
    )
    
    return fig

# Profile presets with market-driven insights
# Profile presets with market-aligned defaults (3×3 grid)
# Profile presets with grounded, analytics-style insights (3×3 grid)
PROFILE_PRESETS = {
    "🎉 Family Party Game": {
        "description": "Light rules, high turn cadence",
        "cats": ["Cat:Family", "Cat:Party"],
        "mechs_on": ["Set Collection", "Dice Rolling"],
        "year": CURRENT_YEAR, "min_players": 3, "max_players": 8,
        "play_time": 30, "min_age": 8, "weight": 1.7,
        "market_insight": "Common profile is 20–40 min at ~1.6–2.0 weight and ≥6 players. Ratings fall off past ~45 min; optimize teach time and turn speed."
    },
    "🤝 Cooperative Strategy": {
        "description": "Team vs game with role clarity",
        "cats": ["Cat:Thematic", "Cat:Strategy"],
        "mechs_on": ["Cooperative Game", "Variable Player Powers", "Action Points"],
        "year": CURRENT_YEAR, "min_players": 1, "max_players": 4,
        "play_time": 60, "min_age": 12, "weight": 2.6,
        "market_insight": "1–4 players, ~2.3–2.8 weight, 45–90 min. Solo support increases reachable audience. Difficulty curves matter more than duration."
    },
    "🏛️ Heavy Euro Game": {
        "description": "Tight economies, scarce actions",
        "cats": ["Cat:Strategy", "Economic"],
        "mechs_on": ["Worker Placement", "Market", "Network and Route Building"],
        "year": CURRENT_YEAR, "min_players": 2, "max_players": 4,
        "play_time": 120, "min_age": 14, "weight": 3.6,
        "market_insight": "Typical lane is 90–150 min at 3.3–3.9 weight. Ratings track visible trade-offs per turn and low upkeep, not rule count."
    },
    "⚔️ Thematic Adventure": {
        "description": "Progression with bounded upkeep",
        "cats": ["Cat:Thematic", "Cat:Adventure"],
        "mechs_on": ["Dice Rolling", "Tile Placement", "Variable Player Powers"],
        "year": CURRENT_YEAR, "min_players": 1, "max_players": 4,
        "play_time": 90, "min_age": 12, "weight": 2.8,
        "market_insight": "1–4 players, 60–120 min, ~2.5–3.0 weight. Progression helps, but bookkeeping is the failure mode; cap it per turn."
    },
    "♟️ Abstract Strategy": {
        "description": "Low luck, perfect information",
        "cats": ["Cat:Abstract"],
        "mechs_on": ["Grid Movement", "Pattern Building", "Area Majority / Influence"],
        "year": CURRENT_YEAR, "min_players": 2, "max_players": 2,
        "play_time": 25, "min_age": 10, "weight": 2.1,
        "market_insight": "2-player, 15–30 min, ~1.8–2.3 weight. Elegance is the lever: compress rules, increase move quality, minimize draw steps."
    },
    "🃏 Deck Builder": {
        "description": "Lean starts, steady upgrades",
        "cats": ["Cat:Card Game", "Cat:Strategy"],
        "mechs_on": ["Deck Construction", "Hand Management", "Set Collection"],
        "year": CURRENT_YEAR, "min_players": 2, "max_players": 4,
        "play_time": 45, "min_age": 10, "weight": 2.3,
        "market_insight": "2–4 players at 30–60 min, ~2.0–2.6 weight. Early turns must be fast; thin starting decks keep tempo and ratings up."
    },
    "🎭 Social Deduction": {
        "description": "Hidden teams, high interaction",
        "cats": ["Cat:Party"],
        "mechs_on": ["Hidden Roles", "Voting", "Player Elimination"],
        "year": CURRENT_YEAR, "min_players": 6, "max_players": 10,
        "play_time": 20, "min_age": 10, "weight": 1.6,
        "market_insight": "6–10 players, 15–30 min, ~1.5–1.8 weight. Limit hard elimination and keep round cadence tight; information flow drives reception."
    },
    "📝 Roll & Write": {
        "description": "Low setup, visible combos",
        "cats": ["Cat:Family", "Cat:Dice"],
        "mechs_on": ["Dice Rolling", "Pattern Building", "Set Collection"],
        "year": CURRENT_YEAR, "min_players": 1, "max_players": 4,
        "play_time": 25, "min_age": 8, "weight": 1.5,
        "market_insight": "1–4 players, 20–30 min, ~1.3–1.7 weight. Variability should come from sheets, not extra rules; minimize idle time."
    },
    "🧩 Tile-Layer Euro": {
        "description": "Spatial scoring, constrained drafting",
        "cats": ["Cat:Strategy", "Cat:City Building"],
        "mechs_on": ["Tile Placement", "Pattern Building", "Set Collection"],
        "year": CURRENT_YEAR, "min_players": 2, "max_players": 4,
        "play_time": 60, "min_age": 10, "weight": 2.3,
        "market_insight": "2–4 players, 45–75 min, ~2.0–2.5 weight. Make scoring visible and cuts obvious; drafting beats bookkeeping for pace."
    }
}

@st.cache_resource(show_spinner=False)
def load_models(paths: Dict[str, str]):
    models = {}
    for key, path in paths.items():
        if not os.path.exists(path):
            continue
        try:
            models[key] = joblib_load(path)
        except Exception:
            pass
    
    if os.path.exists(INPUT_SCALER_PATH):
        try:
            models["_input_scaler"] = joblib_load(INPUT_SCALER_PATH)
        except Exception:
            pass
    
    return models

# Main UI starts here
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("🎲 Board Game Developer Console")
st.subheader("Professional Market Intelligence & Design Analytics Platform")
st.markdown("*Transform market data into actionable design decisions with AI-powered insights*")
st.markdown('</div>', unsafe_allow_html=True)

# Sidebar with enhanced controls
st.sidebar.title("🎮 Control Panel")
st.sidebar.markdown("---")

# Always load from the default dataset paths (no upload control)
df = load_df(None)  # or just load_df() if you set a default arg

# Advanced filtering controls
st.sidebar.markdown("### Analysis Parameters")
k = st.sidebar.slider("# of Clusters", 2, 20, 10, 
                      help="Higher values create more specific market segments")
topn = st.sidebar.slider("Comparison Pool Size", 5, 50, 20,
                         help="Number of similar games to analyze")

st.sidebar.markdown("### Market Filters ")
year_col = "Year Published"
min_year, max_year = int(df[year_col].min()), int(df[year_col].max())
display_min_year = max(1900, min_year)

yr_rng = st.sidebar.slider(
    "📅 Year Range 📅", 
    display_min_year, max_year, 
    (max(1950, display_min_year), max_year),
    help="Focus on specific time periods"
)

weight_col = "GameWeight"
min_w, max_w = float(df[weight_col].min()), float(df[weight_col].max())
wt_rng = st.sidebar.slider(
    "🧩 Complexity Range 🧩", 
    1.0, 5.0, 
    (1.5, 4.0),
    step=0.1,
    help="1=Very Simple, 5=Very Complex"
)

pt_rng_hrs = st.sidebar.slider(
    "⏱️ Play Time (hours) ⏱️", 
    0.25, 6.0, 
    (0.5, 3.0), 
    step=0.25,
    help="Target game duration"
)

pl_min, pl_max = st.sidebar.slider(
    "👥 Player Count Range", 
    1, 12,
    (2, 6),
    help="Supported player counts"
)

age_rng = st.sidebar.slider(
    "👶 Age Range 👴", 
    3, 18, 
    (8, 14),
    help="Minimum age requirements"
)

# Mechanic and theme filtering
with st.sidebar.expander("🎲 Mechanics & Themes 🎲", expanded=False):
    mech_cols = [c for c in df.columns if c.startswith("Mechanic_") or c in [
        "Deck Construction", "Hand Management", "Worker Placement", "Cooperative Game",
        "Dice Rolling", "Set Collection", "Action Points", "Variable Player Powers"
    ]]
    theme_cols = [c for c in df.columns if c.startswith("Cat:") or c in [
        "Fantasy", "Adventure", "Economic", "Science Fiction", "War", "Horror"
    ]]
    
    mech_match_mode = st.radio("Mechanic Match", ["Any", "All"], horizontal=True)
    selected_mechs = st.multiselect("Select Mechanics", mech_cols[:30], default=[])
    
    theme_match_mode = st.radio("Theme Match", ["Any", "All"], horizontal=True)
    selected_themes = st.multiselect("Select Themes", theme_cols[:20], default=[])

st.sidebar.markdown("---")
st.sidebar.caption("💡 **Tip:** Use filters to focus on your target market segment for more accurate insights")
########## narrative toggle + helper ##########
st.sidebar.markdown("### 📝 Narrative")
st.sidebar.checkbox("Show narrative insights", True, key="show_narrative")

def narr(md: str):
    if st.session_state.get("show_narrative", True):
        bl = "> " + md.strip().replace("\n", "\n> ")
        st.markdown(bl)

# Convert play time range to minutes
pt_rng = (int(pt_rng_hrs[0] * 60), int(pt_rng_hrs[1] * 60))

# Prepare data and clustering
X_all, meta = split_features(df)
for _c in ("NumWish","NumWant"):
    if _c in X_all.columns:
        X_all = X_all.drop(columns=[_c])   #don't predict off theses
scaler, kmeans, pca, labels, coords = fit_clusterer(X_all, k=k)

# Add derived features
df["Play Time Hours"] = df["Play Time"] / 60.0
df["Play Time Hours"] = df["Play Time Hours"].clip(upper=10)
df["Success Score"] = (df["AvgRating"] - 5) * df["Owned Users"] / 1000
df["Market Penetration"] = df["Owned Users"] / df["Owned Users"].max()

view = df.copy()
view["Cluster"] = labels
view["PCA1"] = coords[:, 0]
view["PCA2"] = coords[:, 1]

# Apply filters
mask = pd.Series(True, index=view.index)
mask &= view[year_col].between(yr_rng[0], yr_rng[1])
mask &= view[weight_col].between(wt_rng[0], wt_rng[1])
mask &= view["Play Time"].between(pt_rng[0], pt_rng[1])

if all(col in view.columns for col in ["Min Players", "Max Players"]):
    mask &= (view["Max Players"] >= pl_min) & (view["Min Players"] <= pl_max)

if "Min Age" in view.columns:
    mask &= view["Min Age"].between(age_rng[0], age_rng[1])

# Apply mechanic filters
if selected_mechs:
    mech_masks = [view[m] == 1 for m in selected_mechs if m in view.columns]
    if mech_masks:
        mech_combined = pd.concat(mech_masks, axis=1)
        mask &= mech_combined.any(axis=1) if mech_match_mode == "Any" else mech_combined.all(axis=1)

# Apply theme filters  
if selected_themes:
    theme_masks = [view[t] == 1 for t in selected_themes if t in view.columns]
    if theme_masks:
        theme_combined = pd.concat(theme_masks, axis=1)
        mask &= theme_combined.any(axis=1) if theme_match_mode == "Any" else theme_combined.all(axis=1)

view_f = view[mask].copy()
########## build labels for visible (filtered) data ##########
cluster_labels = generate_cluster_labels(view_f)
if view_f.empty:
    st.error("❌ No games match current filters. Please adjust parameters.")
    st.stop()

# Generate cluster insights
cluster_insights = {}
for cid in view_f["Cluster"].unique():
    cluster_data = view_f[view_f["Cluster"] == cid]
    cluster_insights[cid] = {
        "size": len(cluster_data),
        "avg_rating": cluster_data["AvgRating"].mean(),
        "avg_complexity": cluster_data["GameWeight"].mean(),
        "avg_owners": cluster_data["Owned Users"].median(),
        "opportunity_score": calculate_market_opportunity_score(cluster_data),
        "top_game": cluster_data.nlargest(1, "AvgRating")["Name"].iloc[0] if len(cluster_data) > 0 else "N/A"
    }

# Header metrics with enhanced insights
st.markdown("## 📊 Market Overview")
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Total Games", f"{len(view_f):,}")
    delta = len(view_f[view_f["Year Published"] >= CURRENT_YEAR-1])
    st.caption(f"↑ {delta} new this year")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Market Segments", len(cluster_insights))
    st.caption("AI-identified clusters")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    median_rating = view_f["AvgRating"].median()
    st.metric("Median Rating", f"{median_rating:.2f}")
    trend = "↑" if median_rating > 6.5 else "↓"
    st.caption(f"{trend} Market quality trend")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    median_owners = int(view_f["Owned Users"].median())
    st.metric("Median Owners", f"{median_owners:,}")
    st.caption("Market reach indicator")
    st.markdown('</div>', unsafe_allow_html=True)

with col5:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    median_complexity = view_f["GameWeight"].median()
    st.metric("Avg Complexity", f"{median_complexity:.1f}")
    st.caption("Design sophistication")
    st.markdown('</div>', unsafe_allow_html=True)

with col6:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    success_rate = (view_f["AvgRating"] >= 7.0).mean() * 100
    st.metric("Success Rate", f"{success_rate:.1f}%")
    st.caption("Games rated ≥7.0")
    st.markdown('</div>', unsafe_allow_html=True)

# Enhanced tabs with more analysis
tab_intel, tab_wizard, tab_trends, tab_segments, tab_synergies = st.tabs([
    "🎯 Market Intelligence",
    "🧙‍♂️ Design Wizard", 
    "📈 Trend Analysis",
    "🗺️ Segment Explorer",
    "🔗 Mechanic Synergies"
])

# Market Intelligence Tab
with tab_intel:
    st.markdown("## 🎯 Strategic Market Intelligence")
    
    # Market opportunity analysis
    st.markdown("### 🏆 Top Market Opportunities")
    opportunities = sorted(
        cluster_insights.items(),
        key=lambda x: x[1]["opportunity_score"],
        reverse=True
    )[:5]
    
    # ---- Dual-axis grouped bars: Opportunity (%) vs Avg Rating (0–10) ----
    def wrap_label(s: str, max_chars_per_line: int = 18, max_lines: int = 3) -> str:
        words, lines, cur = s.split(), [], ""
        for w in words:
            if len(cur) + len(w) + (1 if cur else 0) <= max_chars_per_line:
                cur = f"{cur} {w}".strip()
            else:
                lines.append(cur); cur = w
                if len(lines) == max_lines - 1:
                    break
        if cur: lines.append(cur)
        used_words = " ".join(lines).split()
        if len(used_words) < len(words) and len(lines) >= max_lines:
            lines[-1] = (lines[-1] + "…").rstrip()
        return "<br>".join(lines)
    
    # Build arrays
    cids = [cid for cid, _ in opportunities]
    labels_raw = [cluster_labels.get(cid, f"Segment {cid}") for cid in cids]
    labels_wrapped = [wrap_label(lbl, 18, 3) for lbl in labels_raw]
    opp_scores = [min(max(cluster_insights[cid]["opportunity_score"], 0), 100) for cid in cids]  # 0–100
    avg_ratings = [min(max(cluster_insights[cid]["avg_rating"], 0), 10) for cid in cids]         # 0–10
    avg_ratings_scaled = [v * 10 for v in avg_ratings]  # scale to 0–100 so bars group
    xs = list(range(len(cids)))  # numeric x positions
    
    fig_opp_bar = go.Figure()
    
    # Opportunity bars (left axis)
    fig_opp_bar.add_trace(go.Bar(
        x=xs,
        y=opp_scores,
        name="Opportunity (%)",
        marker_color=SUCCESS_COLOR,
        opacity=0.9
    ))
    
    # Avg rating bars (scaled; left axis for grouping)
    fig_opp_bar.add_trace(go.Bar(
        x=xs,
        y=avg_ratings_scaled,
        name="Avg Rating (0–10)",
        marker_color=CHART_COLORS[1],
        opacity=0.85
    ))
    
    # Dummy y2 trace so the right axis renders
    fig_opp_bar.add_trace(go.Scatter(
        x=[None], y=[None], yaxis="y2", showlegend=False, hoverinfo="skip"
    ))
    
    # Axes + layout
    fig_opp_bar.update_yaxes(title_text="Opportunity (%)", range=[0, 100])
    fig_opp_bar.update_layout(
        yaxis2=dict(
            title="Avg Rating (0–10)",
            overlaying="y",
            side="right",
            range=[0, 10],
            tickmode="linear",
            dtick=1,
            showgrid=False,
            title_standoff=8
        ),
        barmode="group",
        bargap=0.25,
        bargroupgap=0.05,
        plot_bgcolor=CHART_BG,
        paper_bgcolor=CHART_BG,
        font_color=MUTED,
        legend_title_text="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(t=50, b=110, l=60, r=95)
    )
    
    # Map numeric x back to wrapped labels
    fig_opp_bar.update_xaxes(
        title_text="",
        tickmode="array",
        tickvals=xs,
        ticktext=labels_wrapped,
        tickangle=0,
        automargin=True
    )
    
    st.plotly_chart(fig_opp_bar, use_container_width=True)
# -------------------------------------------------------------------
    # -------------------------------------------------------------------

    opp_cols = st.columns(len(opportunities))
    for i, (cluster_id, data) in enumerate(opportunities):
        with opp_cols[i]:
            st.markdown('<div class="earthcard">', unsafe_allow_html=True)
            st.markdown(f"**{cluster_labels.get(cluster_id, f'Segment {cluster_id}')}**")
            score = data["opportunity_score"]
            color = SUCCESS_COLOR if score > 70 else WARNING_COLOR if score > 40 else DANGER_COLOR
            st.markdown(f"<h2 style='color: {color}'>{score:.0f}%</h2>", unsafe_allow_html=True)
            st.caption(f"Opportunity Score")
            st.write(f"📊 {data['size']} games")
            st.write(f"⭐ {data['avg_rating']:.2f} avg")
            st.write(f"🎯 Example: {data['top_game'][:20]}...")
            st.markdown('</div>', unsafe_allow_html=True)

    narr("""
    **How to use these segments.** Opportunity Score highlights spots where demand is healthy and recent supply is thin. 
    We weight it like this: 3-year release growth 30%, average rating on a 10 point scale 25%, median owners as the audience proxy 20%, 
    low recent saturation 15%, and the share of titles at 7.0 or higher 10%. Strong pockets tend to be mid-weight with real decision space, 60 to 90 minutes, 
    age 10+. Co-op with a solo mode often lifts uptake; people learn alone, then bring it to the table (Spirit Island is a clean example). 
    A small segment that keeps posting solid ratings and owners is a good bet.
    """)

    # Market evolution timeline
    st.markdown("### Complete Market Evolution (1950-Present)")
    evolution_fig = create_market_evolution_timeline(view_f)
    st.plotly_chart(evolution_fig, use_container_width=True)
    narr("""
    **What changed over time.** Since Catan, there was a boardgame boom. 
    Complexity dropped to appeal to more people and then grew as those people got better at games. 
    The average game today beats Catan in ratings illustrating how much better game design has gotten.
    """)

    # Key insights
    st.markdown("### 💡 Data-Driven Market Insights")
    
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("#### Growth Sectors")
        
        # Calculate growth by year
        recent = view_f[view_f["Year Published"] >= CURRENT_YEAR - 3]
        older = view_f[(view_f["Year Published"] >= CURRENT_YEAR - 6) & (view_f["Year Published"] < CURRENT_YEAR - 3)]
        
        if len(recent) > 0 and len(older) > 0:
            growth_rate = (len(recent) / 3) / max(1, len(older) / 3) - 1
            st.write(f"• Market growing at **{growth_rate*100:.1f}%** annually")
            
            # Complexity trends
            recent_complexity = recent["GameWeight"].mean()
            older_complexity = older["GameWeight"].mean()
            complexity_trend = "increasing" if recent_complexity > older_complexity else "decreasing"
            st.write(f"• Game complexity is **{complexity_trend}** ({older_complexity:.2f} → {recent_complexity:.2f})")
            
            # Success metrics
            recent_success = (recent["AvgRating"] >= 7.0).mean()
            older_success = (older["AvgRating"] >= 7.0).mean()
            if recent_success > older_success:
                st.write(f"• Quality improving: **{recent_success*100:.0f}%** recent games are highly rated")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with insight_col2:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("#### Sweet Spots Identified")
        
        # Find optimal characteristics
        top_games = view_f[view_f["AvgRating"] >= 7.5]
        if len(top_games) > 10:
            optimal_weight = top_games["GameWeight"].median()
            optimal_time = top_games["Play Time"].median()
            optimal_players = top_games["Max Players"].mode().iloc[0] if len(top_games["Max Players"].mode()) > 0 else 4
            
            st.write(f"• Optimal complexity: **{optimal_weight:.1f}/5.0**")
            st.write(f"• Ideal play time: **{int(optimal_time)} minutes**")
            st.write(f"• Best player count: **2-{int(optimal_players)} players**")
            
            # Kickstarter impact
            if "Kickstarted" in view_f.columns:
                ks_success = view_f[view_f["Kickstarted"] == 1]["AvgRating"].mean()
                non_ks_success = view_f[view_f["Kickstarted"] == 0]["AvgRating"].mean()
                if ks_success > non_ks_success:
                    st.write(f"• Kickstarter advantage: **+{ks_success - non_ks_success:.2f}** rating points")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Success prediction charts
    st.markdown("### 🎲 Success Factor Analysis")
    
    pred_col1, pred_col2 = st.columns(2)
    
    with pred_col1:
        complexity_success_fig = create_success_predictor_chart(view_f, "GameWeight", "AvgRating")
        st.plotly_chart(complexity_success_fig, use_container_width=True)
    
    with pred_col2:
        time_success_fig = create_success_predictor_chart(view_f, "Play Time Hours", "AvgRating")
        st.plotly_chart(time_success_fig, use_container_width=True)
        
    narr("""
    **Complexity and time, together.** Longer games tend to rate higher, but there is a ceiling on how many people will buy a three hour commitment. 
    Short and complex is the unicorn. When a game is difficult and plays fast, it does well or terribly. If a game is simple, keep it short. Stretching a simple idea for an hour is tough.
    """)

# Design Wizard Tab  
with tab_wizard:
    st.markdown("## 🧙‍♂️ Game Design Wizard 🧙🏾‍♂️")
    st.markdown("*Build your concept and receive market-validated predictions and recommendations*")
    
    # Archetype selection with insights
    st.markdown("### 🎮 Choose Your Design Archetype")
    
    archetype_cols = st.columns(3)
    selected_preset = None
    
    for i, (preset_name, preset_data) in enumerate(PROFILE_PRESETS.items()):
        with archetype_cols[i % 3]:
            if st.button(preset_name, key=f"preset_{i}", use_container_width=True):
                selected_preset = preset_name
            st.caption(preset_data["description"])
            st.info(f"💡 {preset_data['market_insight']}")
    
    if selected_preset:
        st.session_state.selected_preset = selected_preset
    
    current_preset = st.session_state.get("selected_preset", list(PROFILE_PRESETS.keys())[0])
    preset_data = PROFILE_PRESETS[current_preset]
    
    st.markdown(f"### 📝 Designing: {current_preset}")
    
    with st.form("design_form"):
        st.markdown("#### Core Specifications")
        spec_cols = st.columns(4)
        with spec_cols[0]:
            year_published = st.number_input("Release Year (2022 data incomplete, assumes 2021)", value=preset_data["year"],
                                             min_value=CURRENT_YEAR-30, max_value=CURRENT_YEAR+1, key="year_input")
            min_players = st.number_input("Min Players", value=preset_data["min_players"],
                                          min_value=1, max_value=10, key="min_players_input")
        with spec_cols[1]:
            play_time = st.number_input("Play Time (min)", value=preset_data["play_time"],
                                        min_value=5, max_value=360, step=5, key="playtime_input")
            max_players = st.number_input("Max Players", value=preset_data["max_players"],
                                          min_value=min_players, max_value=20, key="max_players_input")
        with spec_cols[2]:
            complexity = st.slider("Complexity", 1.0, 5.0, preset_data["weight"], 0.1, key="complexity_input")
            min_age = st.number_input("Min Age", value=preset_data["min_age"], min_value=3, max_value=21, key="min_age_input")
        with spec_cols[3]:
            kickstarted = st.selectbox("Funding Model", ["Traditional", "Kickstarter", "Gamefound"], index=0, key="funding_input")
            production_quality = st.select_slider("Production Quality", ["Basic", "Standard", "Premium"],
                                                  value="Standard", key="prod_quality_input")
    
        # --- Mechanics & Themes (stacked) ---
        with st.expander("Mechanics & Themes (toggle selections)", expanded=True):
            # Mechanics (Solo + Co-op)
            st.markdown("##### Mechanics")
            solo_mode = st.checkbox("Solo Mode", value=(preset_data.get("min_players", 1) == 1), key="solo_mode_toggle")
            coop_toggle = st.checkbox("Co-op (team vs game)",
                                      value=("Cooperative Game" in preset_data.get("mechs_on", [])),
                                      key="coop_toggle")
    
            mech_all = discover_mechanics(X_all.columns)
            # remove Co-op from grid; it's its own toggle
            mech_all = [m for m in mech_all if m not in {"Cooperative Game", "Mechanic_Cooperative_Game"}]
            mech_defaults = [
                m if m in mech_all else f"Mechanic_{m.replace(' ', '_')}"
                for m in preset_data.get("mechs_on", [])
                if (m in mech_all or f"Mechanic_{m.replace(' ', '_')}" in mech_all)
            ]
            selected_mechanics = render_toggle_grid(mech_all, defaults=mech_defaults, columns=3, key_prefix="mech")
    
            st.markdown("---")
    
            # Themes (under mechanics)
            st.markdown("##### Themes & Categories")
            theme_all = discover_themes(X_all.columns)
            theme_defaults = [c for c in preset_data.get("cats", []) if c in theme_all]
            selected_themes = render_toggle_grid(theme_all, defaults=theme_defaults, columns=3, key_prefix="theme")
    
        # --- Pricing & Unit Economics (moved inside form) ---
        st.markdown("#### Pricing & Unit Economics")
        with st.expander("Set price & cost assumptions", expanded=True):
            r1c1, r1c2, r1c3, r1c4 = st.columns(4)
            with r1c1:
                target_price = st.slider("Target MSRP ($)", 20, 150, 50, 1, key="msrp_input")
            with r1c2:
                component_quality = st.select_slider("Component Quality",
                                                     ["Basic", "Good", "Premium"],
                                                     value="Good", key="component_quality_input")
            with r1c3:
                sales_window = st.slider("Sales Window (months)", 3, 36, 12, key="sales_window_input")
            with r1c4:
                returns_pct = st.slider("Returns/Damage Allowance (%)", 0, 20, 5, key="returns_input") / 100.0
    
            r2c1, r2c2, r2c3, r2c4 = st.columns(4)
            with r2c1:
                unit_cogs = st.slider("Unit Cost to Produce ($)", 1, 60, 12, key="cogs_input")
            with r2c2:
                shipping_per_unit = st.slider("Fulfillment & Shipping per Unit ($)", 0, 30, 5, key="ship_input")
            with r2c3:
                marketing_fixed = st.number_input("Marketing Budget ($, fixed)", 0, 500_000, 25_000, step=1_000, key="mkt_input")
            with r2c4:
                misc_fixed = st.number_input("Misc & Dev ($, fixed)", 0, 500_000, 15_000, step=1_000, key="misc_input")
    
            with st.expander("Advanced assumptions", expanded=False):
                fee_default = default_channel_fee_pct(kickstarted)
                channel_fee_pct = st.slider("Retailer / Platform Fee (%)", 0, 70, int(fee_default*100),
                                            key="fee_input") / 100.0
                apply_sensitivity = st.checkbox("Apply price elasticity to owners", value=True, key="elasticity_toggle")
                elasticity = st.slider("Price Elasticity (negative)", -2.0, -0.1,
                                       -1.1 if kickstarted == "Traditional" else -0.8, 0.1,
                                       key="elasticity_input")
    
        # >>> Submit button must be inside the form <<<
        analyze_button = st.form_submit_button("🔮 Analyze Design & Generate Predictions", type="primary", use_container_width=True)
    
    if analyze_button:
        # Build comprehensive profile
        profile = {
            "Year Published": year_published,
            "Min Players": min_players,
            "Max Players": max_players,
            "Play Time": play_time,
            "Min Age": min_age,
            "GameWeight": complexity,
            "Kickstarted": 1 if kickstarted != "Traditional" else 0,
        }
        
        # Add mechanics and themes
        # Solo Mode → force Min Players = 1 (model-safe)
        if solo_mode:
            profile["Min Players"] = 1
        
        # Co-op toggle → map to real column if present
        toggle_feature(profile, X_all.columns, coop_toggle, "Cooperative Game")
        
        # Mechanics toggles (exact column names from grid)
        for m in selected_mechanics:
            profile[m] = 1
        
        # Themes toggles (exact column names from grid)
        for t in selected_themes:
            profile[t] = 1
        
        
        # Prepare input for clustering
        x_input = pd.DataFrame([{c: profile.get(c, 0) for c in X_all.columns}])
        x_scaled = scaler.transform(x_input)
        cluster_id = int(kmeans.predict(x_scaled)[0])
        
        # Find similar games
        cluster_games = view_f[view_f["Cluster"] == cluster_id]
        
        if len(cluster_games) > 0:
            # Calculate distances for nearest neighbors
            X_cluster = X_all.loc[cluster_games.index]
            X_cluster_scaled = scaler.transform(X_cluster)
            distances = pairwise_distances(x_scaled, X_cluster_scaled)[0]
            cluster_games = cluster_games.copy()
            cluster_games["__dist"] = distances
            neighbors = cluster_games.nsmallest(min(topn, len(cluster_games)), "__dist")
            
            # Generate predictions
            st.markdown("---")
            st.markdown("## 👨🏾‍🔬 Design Analysis Results")
            
            # AI Predictions section
            st.markdown("### 🤖 AI Performance Predictions")
            # --- Predictions & Rendering (robust to missing models) ---
            FEATURE_COLS_PATH = "models/feature_cols.json"  # define this once
            
            models = load_models(MODEL_PATHS)
            use_models = any(k in models for k in ("rating_xgb", "owned_rf", "owned_xgb"))
            
            # Guard: if the currently selected cluster has no games (due to tight filters),
            # fall back to nearest neighbors across ALL filtered games.
            if len(cluster_games) == 0:
                st.warning("No games in the selected segment under current filters. Falling back to all filtered games.")
                X_cluster = X_all.loc[view_f.index]
                X_cluster_scaled = scaler.transform(X_cluster)
                distances = pairwise_distances(x_scaled, X_cluster_scaled)[0]
                tmp = view_f.copy()
                tmp["__dist"] = distances
                neighbors = tmp.nsmallest(min(topn, len(tmp)), "__dist")
            else:
                # Normal neighbor selection within the predicted cluster
                X_cluster = X_all.loc[cluster_games.index]
                X_cluster_scaled = scaler.transform(X_cluster)
                distances = pairwise_distances(x_scaled, X_cluster_scaled)[0]
                cluster_games = cluster_games.copy()
                cluster_games["__dist"] = distances
                neighbors = cluster_games.nsmallest(min(topn, len(cluster_games)), "__dist")
            
            if len(neighbors) == 0:
                st.error("Couldn’t find similar games to compare against. Loosen filters and try again.")
                st.stop()
            
            # === Compute predictions (models if available, otherwise neighbor fallback) ===
            if use_models:
                scaler_in = models.get("_input_scaler")
                training_cols = None
            
                # 1) Try to load saved feature ordering from JSON
                try:
                    if os.path.exists(FEATURE_COLS_PATH):
                        import json
                        with open(FEATURE_COLS_PATH, "r", encoding="utf-8") as f:
                            training_cols = [c for c in json.load(f) if c not in PRED_EXCLUDE]
                except Exception:
                    training_cols = None
            
                # 2) Else use scaler’s or model’s feature names
                if training_cols is None and scaler_in is not None and hasattr(scaler_in, "feature_names_in_"):
                    training_cols = [c for c in scaler_in.feature_names_in_ if c not in PRED_EXCLUDE]
                if training_cols is None:
                    for _k in ("rating_xgb", "owned_rf", "owned_xgb"):
                        mdl = models.get(_k)
                        if mdl is not None and hasattr(mdl, "feature_names_in_"):
                            training_cols = [c for c in mdl.feature_names_in_ if c not in PRED_EXCLUDE]
                            break
                if training_cols is None:
                    training_cols = [c for c in X_all.columns if c not in PRED_EXCLUDE]
            
                # 3) Align, optional scale, predict
                X_pred_raw = align_profile_to_training(profile, training_cols, scaler=None)
                X_in = X_pred_raw
                if scaler_in is not None:
                    try:
                        X_scaled_cols = scaler_in.transform(X_pred_raw.values)
                        X_in = pd.DataFrame(X_scaled_cols, columns=training_cols)
                    except Exception:
                        X_in = X_pred_raw
            
                rating_model = models.get("rating_xgb") or models.get("rating") or models.get("rating_model")
                if rating_model is not None:
                    try:
                        predicted_rating = float(np.clip(_predict_agnostic(rating_model, X_in, training_cols), 0.0, 10.0))
                    except Exception:
                        predicted_rating = float(neighbors["AvgRating"].mean())
                else:
                    predicted_rating = float(neighbors["AvgRating"].mean())
            
                owners_model = models.get("owned_rf") or models.get("owned_xgb")
                if owners_model is not None:
                    try:
                        owners_val = _predict_agnostic(owners_model, X_in, training_cols)
                        predicted_owners = int(max(0, owners_val))
                    except Exception:
                        predicted_owners = int(neighbors["Owned Users"].median())
                else:
                    predicted_owners = int(neighbors["Owned Users"].median())
            
            else:
                # --- Fallback using neighbors ---
                # Deterministic, distance-weighted fallback (no randomness)
                neighbors = neighbors.sort_values(
                    ["__dist", "BGGId" if "BGGId" in neighbors.columns else "Name"]
                ).head(min(topn, len(neighbors)))
                
                w = 1.0 / (1e-9 + neighbors["__dist"].astype(float))
                w = w / w.sum()
                
                predicted_rating = float(np.clip((neighbors["AvgRating"].astype(float) * w).sum(), 0.0, 10.0))
                predicted_owners = int(max(0, (neighbors["Owned Users"].astype(float) * w).sum()))
                
            ########## soft calibration (quick testing only) ##########
            # stand in for low results because of empty vectors above. use nearest neghbors, boost ownership for good ratings
            # def _interp(x, a, b, c, d):
            #     # linear map [a,b] -> [c,d] with clamping
            #     if b <= a: 
            #         return float(np.clip((c + d) / 2.0, c, d))
            #     t = (x - a) / (b - a)
            #     return float(c + np.clip(t, 0.0, 1.0) * (d - c))
            
            # # Use neighbor distribution to set source ranges; fallback to sensible defaults
            # if len(neighbors) >= 5:
            #     r_src_lo = float(neighbors["AvgRating"].quantile(0.25))
            #     r_src_hi = float(neighbors["AvgRating"].quantile(0.75))
            #     o_src_lo = float(neighbors["Owned Users"].quantile(0.25))
            #     o_src_hi = float(neighbors["Owned Users"].quantile(0.90))
            # else:
            #     # If neighbors are thin, assume your observed behavior (5.5–6.5 ratings, few hundred owners)
            #     r_src_lo, r_src_hi = 5.5, 6.5
            #     o_src_lo, o_src_hi = 200.0, 3000.0
            
            # # Target bands 
            # r_dst_lo, r_dst_hi = 6.0, 8.0
            # o_dst_lo, o_dst_hi = 500.0, 20000.0
            
            # # Map predicted_rating and owners into those bands (with clamps)
            # predicted_rating = _interp(predicted_rating, r_src_lo, r_src_hi, r_dst_lo, r_dst_hi)
            # predicted_rating = float(np.clip(predicted_rating, 5.0, 9.2))  # hard safety clamp
            
            # predicted_owners = _interp(float(predicted_owners), o_src_lo, o_src_hi, o_dst_lo, o_dst_hi)
            
            # # Small bonus: let higher ratings nudge owners a bit (keeps things consistent)
            # owners_rating_factor = _interp(predicted_rating, r_dst_lo, r_dst_hi, 0.9, 1.15)
            # predicted_owners = int(np.round(np.clip(predicted_owners * owners_rating_factor, 100, 50000)))*0.4

###################end stand in block ####################################
           # === Compute predictions (prefer trained models; fallback to nearest neighbors) ===
            predicted_rating = None
            predicted_owners = None
            
            try:
                # 0) Models already loaded above:
                # models = load_models(MODEL_PATHS)
            
                # 1) Resolve training feature order
                scaler_in = models.get("_input_scaler")
                training_cols = None
                FEATURE_COLS_PATH = "models/feature_cols.json"
            
                # 1a) JSON list (preferred)
                try:
                    if os.path.exists(FEATURE_COLS_PATH):
                        with open(FEATURE_COLS_PATH, "r", encoding="utf-8") as f:
                            training_cols = [c for c in json.load(f) if c not in PRED_EXCLUDE]
                except Exception:
                    training_cols = None
            
                # 1b) Scaler feature names
                if training_cols is None and scaler_in is not None and hasattr(scaler_in, "feature_names_in_"):
                    training_cols = [c for c in scaler_in.feature_names_in_ if c not in PRED_EXCLUDE]
            
                # 1c) Model feature names
                if training_cols is None:
                    for _k in ("rating_xgb", "owned_rf", "owned_xgb"):
                        mdl = models.get(_k)
                        if mdl is not None and hasattr(mdl, "feature_names_in_"):
                            training_cols = [c for c in mdl.feature_names_in_ if c not in PRED_EXCLUDE]
                            break
            
                # 1d) Fall back to current X_all
                if training_cols is None:
                    training_cols = [c for c in X_all.columns if c not in PRED_EXCLUDE]
            
                # 2) Align and optionally scale
                X_pred_raw = align_profile_to_training(profile, training_cols, scaler=None)
                X_in = X_pred_raw
                if scaler_in is not None:
                    try:
                        X_scaled_cols = scaler_in.transform(X_pred_raw.values)
                        X_in = pd.DataFrame(X_scaled_cols, columns=training_cols)
                    except Exception:
                        X_in = X_pred_raw
            
                # 3) Get models; require both
                rating_model = models.get("rating_xgb") or models.get("rating") or models.get("rating_model")
                owners_model = models.get("owned_rf") or models.get("owned_xgb")
                if rating_model is None or owners_model is None:
                    raise RuntimeError("Missing rating and/or owners model")
            
                # 4) Predict (raw model outputs; no heuristic remaps)
                predicted_rating = float(np.clip(_predict_agnostic(rating_model, X_in, training_cols), 0.0, 10.0))
                owners_val = float(_predict_agnostic(owners_model, X_in, training_cols))
                predicted_owners = int(max(0, round(owners_val)))
            
            except Exception as e:
                # Neighbor fallback (kept intact)
                st.warning(f"Model prediction unavailable, using nearest-neighbor fallback: {e}")
                neighbors_sorted = neighbors.sort_values(
                    ["__dist", "BGGId" if "BGGId" in neighbors.columns else "Name"]
                ).head(min(topn, len(neighbors)))
            
                w = 1.0 / (1e-9 + neighbors_sorted["__dist"].astype(float))
                w = w / w.sum()
            
                predicted_rating = float(np.clip((neighbors_sorted["AvgRating"].astype(float) * w).sum(), 0.0, 10.0))
                predicted_owners = int(max(0, (neighbors_sorted["Owned Users"].astype(float) * w).sum()))

            
          
            # ---------- SMART SUCCESS PROBABILITY (rating + owners + fit + alignment) ----------
            def _sig(x: float) -> float:
                return 1.0 / (1.0 + np.exp(-x))
            
            def _interp(x, a, b, c, d):
                # linear map [a,b] -> [c,d] with clamping
                if b <= a:
                    return float((c + d) * 0.5)
                t = (x - a) / (b - a)
                return float(c + np.clip(t, 0.0, 1.0) * (d - c))
            
            def _q(series, q, default):
                try:
                    v = float(np.quantile(pd.to_numeric(series, errors="coerce").dropna().values, q))
                    return v if np.isfinite(v) else default
                except Exception:
                    return default
            
            # Use cluster as reference if it’s not tiny; else fall back to all filtered
            ref = cluster_games if len(cluster_games) >= 30 else view_f
            
            # 1) Rating score (blend of percentile and a sigmoid around 7.0)
            r_med = _q(ref["AvgRating"], 0.50, 6.6)
            r_p90 = _q(ref["AvgRating"], 0.90, 7.7)
            rating_pct = stats.percentileofscore(ref["AvgRating"], predicted_rating)  # keep for UI
            rating_lin = np.clip((predicted_rating - r_med) / max(1e-6, r_p90 - r_med), 0, 1)
            rating_sig = _sig((predicted_rating - 7.0) / 0.35)
            rating_score = 0.5 * rating_lin + 0.5 * rating_sig  # 0–1
            
            # 2) Owners score (log-scaled vs median→p90 band)
            o_med = _q(ref["Owned Users"], 0.50, 1200.0)
            o_p90 = _q(ref["Owned Users"], 0.90, 15000.0)
            owners_score = np.clip(
                (np.log1p(max(0.0, float(predicted_owners))) - np.log1p(o_med)) /
                max(1e-6, (np.log1p(o_p90) - np.log1p(o_med))),
                0, 1
            )
            
            # 3) Fit score (how tight your neighbors are)
            dvals = neighbors["__dist"].astype(float).values
            fit_score = float(np.clip(1.0 - (dvals.mean() / (dvals.max() + 1e-9)), 0.0, 1.0))
            
            # 4) Alignment score (complexity/time/price vs segment)
            cluster_complexity = float(ref["GameWeight"].median())
            cluster_time = float(ref["Play Time"].median())
            
            comp_align = np.clip(1.0 - abs(float(complexity) - cluster_complexity) / 0.7, 0.0, 1.0)
            time_align = np.clip(1.0 - abs(float(play_time) - cluster_time) / 60.0, 0.0, 1.0)
            
            anchor = estimate_anchor_price(
                float(complexity), component_quality, production_quality, int(max_players), int(play_time)
            )
            price_align = np.clip(1.0 - abs(float(target_price) - anchor) / max(15.0, float(anchor)), 0.0, 1.0)
            
            align_score = 0.5 * comp_align + 0.3 * time_align + 0.2 * price_align  # 0–1
            
            # Weighted blend → raw success (0–1)
            raw_success = (
                0.40 * rating_score +
                0.30 * owners_score +
                0.15 * fit_score +
                0.15 * align_score
            )
            
            # Smooth to % with gentle S-curve and calibrated bounds
            confidence = int(np.round(_interp(_sig((raw_success - 0.55) / 0.18), 0, 1, 30, 97)))
            percentile = rating_pct  # keep existing UI text working
################## smart market size section ############################
            def market_size_label(predicted_owners: int, ref_df: pd.DataFrame, cluster_df: Optional[pd.DataFrame] = None):
                """
                Return (label, percentile, cutoffs) for ownership market size.
                Uses log-space quantiles from the active segment if it's large enough,
                else falls back to all filtered data, else fixed cutoffs.
                """
                # Prefer current segment if it has enough samples; else all filtered
                ref = cluster_df if (cluster_df is not None and len(cluster_df) >= 50) else ref_df
                owners_series = pd.to_numeric(ref.get("Owned Users", pd.Series([])), errors="coerce").dropna()
            
                if len(owners_series) >= 50:
                    # log-space to avoid “whales” blowing up thresholds
                    qs_log = np.quantile(np.log1p(owners_series.values), [0.20, 0.40, 0.60, 0.80])
                    cutoffs = [int(np.expm1(x)) for x in qs_log]  # [Q20, Q40, Q60, Q80]
                elif len(owners_series) >= 10:
                    # simple linear quantiles if sample is small but usable
                    qs = np.quantile(owners_series.values, [0.25, 0.50, 0.75, 0.90])
                    cutoffs = [int(q) for q in qs]
                else:
                    # sane defaults for board games if we basically have no data
                    cutoffs = [1_000, 5_000, 20_000, 50_000]
            
                x = int(max(0, predicted_owners))
                if x < cutoffs[0]:
                    label = "Micro"
                elif x < cutoffs[1]:
                    label = "Niche"
                elif x < cutoffs[2]:
                    label = "Mid-Market"
                elif x < cutoffs[3]:
                    label = "Large Market"
                else:
                    label = "Mass Market"
            
                pct = None
                if len(owners_series) > 0:
                    pct = float(stats.percentileofscore(owners_series.values, x))
            
                return label, pct, cutoffs

            market_size, owners_pct, cutoffs = market_size_label(
                        predicted_owners,
                        ref_df=view_f,
                        cluster_df=cluster_games if 'cluster_games' in locals() else None)
            
            # === Always render the prediction cards (regardless of models present) ===
            pred_cols = st.columns(3)
            
            with pred_cols[0]:
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                st.metric("Predicted Rating", f"{predicted_rating:.2f}/10")
                st.progress(predicted_rating / 10)
                st.caption(f"Top {100 - percentile:.0f}% percentile")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with pred_cols[1]:
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                st.metric("Expected Owners", f"{predicted_owners:,}")
                st.caption(f"📊 {market_size} potential")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with pred_cols[2]:
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                st.metric("Success Probability", f"{confidence}%")
                st.progress(confidence / 100)
                risk_level = "Low Risk" if confidence > 70 else "Moderate Risk" if confidence > 50 else "High Risk"
                st.caption(f"⚠️ {risk_level}")
                st.markdown('</div>', unsafe_allow_html=True)
            

            narr(
                "**Reading your forecast.** Treat these only as a prediction, things to consider: "
                "If the model likes your rating but owners look light, the design might be niche or overpriced. "
                "If owners look strong but rating is middling, you might be making a bad but marketable game"
                "Price can move demand, but not forever. Anchor the MSRP to what the experience feels like in the first 15 minutes."
            )
            
            ########## Pricing & Unit Economics ##########
            st.markdown("### 💵 Economics and pricing 💵")
            
            # Use inputs captured in the form
            msrp = float(target_price)
            anchor_price = estimate_anchor_price(
                complexity, component_quality, production_quality, max_players, play_time
            )
            
            # --- Simple sales estimate: owners minus returns ---
            owners_base = float(predicted_owners)                   # “reported owners” / model output
            owners_adj  = owners_base                                # keep var for downstream compatibility
            effective_units = owners_base * (2.0 - float(returns_pct))  # SALES = owners × (1 - returns)
            
            # Unit economics (unchanged)
            net_per_unit = float(target_price) * (1 - float(channel_fee_pct))
            gross_profit_per_unit = net_per_unit - (float(unit_cogs) + float(shipping_per_unit))
            fixed_costs = float(marketing_fixed + misc_fixed)
            
            total_gross_profit = gross_profit_per_unit * max(effective_units, 0)
            net_profit = total_gross_profit - fixed_costs
            roi_multiple = (net_profit / fixed_costs) if fixed_costs > 0 else float("inf")
            
            breakeven_units = (fixed_costs / gross_profit_per_unit) if gross_profit_per_unit > 0 else float("inf")
            monthly_units = effective_units / max(sales_window, 1)
            payback_months = math.ceil(breakeven_units / max(monthly_units, 1)) if math.isfinite(breakeven_units) else None
            gross_margin_pct = (gross_profit_per_unit / net_per_unit) if net_per_unit > 0 else 0.0
            
            # Metrics row continues...
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            with m1:
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            
                reported_owners = int(round(owners_base))      # before returns
                sales_estimate  = int(round(effective_units))  # after returns (this is your sales)
            
                delta_abs = sales_estimate - reported_owners
                delta_pct = (sales_estimate / reported_owners - 1.0) if reported_owners > 0 else 0.0
            
                st.metric(
                    "Sales estimate (assumes 50% reported owners minus returns)",
                    f"{sales_estimate:,}",
                    delta=f"{delta_abs:+,} ({delta_pct:+.0%})",
                    delta_color="normal"
                )
            
                st.caption(f"Reported owners: {reported_owners:,} • returns {int(returns_pct*100)}%")
                st.markdown('</div>', unsafe_allow_html=True)

            with m2:
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                st.metric("Gross Margin", f"{gross_margin_pct*100:.0f}%/unit")
                st.caption(f"Unit GP ${gross_profit_per_unit:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
            with m3:
                beu_text = f"{int(breakeven_units):,}" if math.isfinite(breakeven_units) and breakeven_units >= 0 else "N/A"
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                st.metric("Break-even Units", beu_text)
                st.caption("Fixed costs ÷ unit gross profit")
                st.markdown('</div>', unsafe_allow_html=True)
            with m4:
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                st.metric("Payback", f"{payback_months or 'N/A'} months")
                st.caption(f"Sales window {sales_window} months")
                st.markdown('</div>', unsafe_allow_html=True)
            with m5:
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                st.metric("Net Profit", f"${int(net_profit):,}")
                st.caption("After fixed costs")
                st.markdown('</div>', unsafe_allow_html=True)
            with m6:
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                roi_disp = "∞" if roi_multiple == float("inf") else f"{roi_multiple:.1f}x"
                st.metric("ROI", roi_disp)
                st.caption("Net profit ÷ fixed costs")
                st.markdown('</div>', unsafe_allow_html=True)
                
            ########## design analysis visuals ##########
            st.markdown("### Your Game and It's Cluster")
            
            # A) Rating vs Complexity (your design highlighted)
            fig_a = go.Figure()
            
            rest = view_f
            seg  = cluster_games
            
            # base cloud (muted)
            fig_a.add_trace(go.Scatter(
                x=rest["GameWeight"], y=rest["AvgRating"],
                mode="markers", name="All filtered games",
                marker=dict(size=5, opacity=0.25),
                hoverinfo="skip"
            ))
            
            # segment highlight
            # segment highlight (ORANGE)
            fig_a.add_trace(go.Scatter(
                x=seg["GameWeight"], y=seg["AvgRating"],
                mode="markers",
                name=cluster_labels.get(cluster_id, f"Segment {cluster_id}"),
                marker=dict(
                    size=8,
                    opacity=0.85,
                    color=SECONDARY,                     # 🔶 make segment dots orange
                    line=dict(width=1, color="rgba(0,0,0,0.25)")
                ),
                text=seg.get("Name", None),
                hovertemplate="<b>%{text}</b><br>Weight: %{x:.2f}<br>Rating: %{y:.2f}<extra></extra>"
            ))
            
            # your game
            fig_a.add_trace(go.Scatter(
                x=[complexity], y=[predicted_rating],
                mode="markers+text", name="Your game",
                marker=dict(symbol="star", size=18, line=dict(width=1), color="red"),
                text=["★"], textposition="top center",
                hovertemplate=f"<b>Your design</b><br>Weight: {complexity:.2f}<br>Pred rating: {predicted_rating:.2f}<extra></extra>"
            ))
            
            fig_a.update_layout(
                title="Rating vs Complexity (Your design highlighted)",
                xaxis_title="Complexity (weight 1–5)",
                yaxis_title="Average Rating",
                paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG, height=420
            )
            st.plotly_chart(fig_a, use_container_width=True)
            
            # B) Rating vs Owners (your expected owners)
            fig_b = go.Figure()
            
            fig_b.add_trace(go.Scatter(
                x=rest["AvgRating"], y=rest["Owned Users"],
                mode="markers", name="All filtered games",
                marker=dict(size=5, opacity=0.25),
                hoverinfo="skip"
            ))
            
            # segment highlight (ORANGE)
            fig_b.add_trace(go.Scatter(
                x=seg["AvgRating"], y=seg["Owned Users"],
                mode="markers",
                name=cluster_labels.get(cluster_id, f"Segment {cluster_id}"),
                marker=dict(
                    size=8,
                    opacity=0.85,
                    color=SECONDARY,                     # 🔶 make segment dots orange
                    line=dict(width=1, color="rgba(0,0,0,0.25)")
                ),
                text=seg.get("Name", None),
                hovertemplate="<b>%{text}</b><br>Rating: %{x:.2f}<br>Owners: %{y:,}<extra></extra>"
            ))

            
            fig_b.add_trace(go.Scatter(
                x=[predicted_rating], y=[predicted_owners],
                mode="markers+text", name="Your game (expected owners)",
                marker=dict(symbol="star", size=18, line=dict(width=1), color="red"),
                text=[f"★ {predicted_owners:,}"], textposition="bottom center",
                hovertemplate=f"<b>Your design</b><br>Pred rating: {predicted_rating:.2f}"
                              f"<br>Expected owners: {predicted_owners:,}<extra></extra>"
            ))
            
            fig_b.update_layout(
                title="Rating vs Owners (Your expected owners highlighted)",
                xaxis_title="Average Rating",
                yaxis_title="Owners (log)",
                yaxis_type="log",
                paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG, height=420
            )
            st.plotly_chart(fig_b, use_container_width=True)
            
            # C) Year vs Rating with segment highlight and your symbol
            fig_c = go.Figure()
            
            fig_c.add_trace(go.Scatter(
                x=rest["Year Published"], y=rest["AvgRating"],
                mode="markers", name="All filtered games",
                marker=dict(size=5, opacity=0.15),
                hoverinfo="skip"
            ))
            
            # segment highlight (ORANGE)
            fig_c.add_trace(go.Scatter(
                x=seg["Year Published"], y=seg["AvgRating"],
                mode="markers",
                name=cluster_labels.get(cluster_id, f"Segment {cluster_id}"),
                marker=dict(
                    size=8,
                    opacity=0.85,
                    color=SECONDARY,                     # 🔶 make segment dots orange
                    line=dict(width=1, color="rgba(0,0,0,0.25)")
                ),
                text=seg.get("Name", None),
                hovertemplate="<b>%{text}</b><br>Year: %{x}<br>Rating: %{y:.2f}<extra></extra>"
            ))

            
            fig_c.add_trace(go.Scatter(
                x=[year_published], y=[predicted_rating],
                mode="markers+text", name="Your game",
                marker=dict(symbol="star", size=20, line=dict(width=1), color="red"),
                text=["★"], textposition="middle right",
                hovertemplate=f"<b>Your design</b><br>Year: {year_published}<br>Pred rating: {predicted_rating:.2f}<extra></extra>"
            ))
            
            fig_c.update_layout(
                title="Year vs Rating (Segment highlighted; your game marked)",
                xaxis_title="Year Published",
                yaxis_title="Average Rating",
                paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG, height=420
            )
            st.plotly_chart(fig_c, use_container_width=True)

            # Market positioning
            st.markdown("### 📍 Market Positioning Analysis")
            
            st.markdown('<div class="earthcard">', unsafe_allow_html=True)
            
            position_cols = st.columns(3)
            
            with position_cols[0]:
                st.markdown("#### 🎯 Competitive Position")
                cluster_complexity = cluster_games["GameWeight"].median()
                if complexity > cluster_complexity + 0.3:
                    st.warning("⚠️ **Higher complexity** than market segment average")
                    st.write("Consider simplifying or targeting hardcore audience")
                elif complexity < cluster_complexity - 0.3:
                    st.success("✅ **More accessible** than competitors")
                    st.write("Good position for market expansion")
                else:
                    st.info("✓ **Well-aligned** with market expectations")
                    st.write("Focus on unique mechanics/theme for differentiation")
            
            with position_cols[1]:
                st.markdown("#### ⏱️ Duration Analysis")
                cluster_time = cluster_games["Play Time"].median()
                if play_time > cluster_time + 30:
                    st.warning("⚠️ **Longer than typical**")
                    st.write("Ensure gameplay justifies extended time")
                elif play_time < cluster_time - 30:
                    st.success("✅ **Quick to play**")
                    st.write("Appeals to time-conscious gamers")
                else:
                    st.info("✓ **Standard duration**")
                    st.write("Meets market expectations")
            
            with position_cols[2]:
                st.markdown("#### 💵 Price Positioning")
                if target_price > 80:
                    st.warning("⚠️ **Premium pricing**")
                    st.write("Requires exceptional components/gameplay")
                elif target_price < 40:
                    st.success("✅ **Accessible price point**")
                    st.write("Good for market penetration")
                else:
                    st.info("✓ **Market-standard pricing**")
                    st.write("Competitive with similar games")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Specific recommendations
            st.markdown("### 💡 Data-Driven Design Recommendations")
            
            recommendations = generate_design_recommendations(cluster_games, view_f)
            
            st.markdown('<div class="recommendation-card">', unsafe_allow_html=True)
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Similar games analysis
            st.markdown("### 🔍 Most Similar Existing Games")
            st.write(f"Found **{len(neighbors)}** highly similar games in the database:")
            
            # Enhanced neighbor display
            neighbor_display = neighbors[["Name", "Year Published", "AvgRating", "Owned Users", 
                                         "GameWeight", "Play Time"]].copy()
            neighbor_display["Success Score"] = (neighbor_display["AvgRating"] - 6) * neighbor_display["Owned Users"] / 1000
            neighbor_display = neighbor_display.sort_values("Success Score", ascending=False)
            
            st.dataframe(
                neighbor_display.head(10).style.background_gradient(subset=["AvgRating", "Owned Users"]),
                use_container_width=True,
                hide_index=True
            )
            
            # Visual comparison
            st.markdown("### 📊 Visual Market Comparison")
            
            comp_col1, comp_col2 = st.columns(2)
            
            with comp_col1:
                # Radar chart comparing to segment average
                categories = ['Complexity', 'Play Time (h)', 'Player Count', 'Age', 'Price']
                
                fig_radar = go.Figure()
                
                # Your game
                your_values = [
                    complexity/5,
                    play_time/180,  # Normalize to 0-1 (180 min max)
                    max_players/10,
                    min_age/18,
                    target_price/150
                ]
                
                # Segment average
                segment_values = [
                    cluster_games["GameWeight"].mean()/5,
                    cluster_games["Play Time"].mean()/180,
                    cluster_games["Max Players"].mean()/10 if "Max Players" in cluster_games else 0.4,
                    cluster_games["Min Age"].mean()/18 if "Min Age" in cluster_games else 0.5,
                    0.33  # Assumed average price position
                ]
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=your_values,
                    theta=categories,
                    fill='toself',
                    name='Your Design',
                    line_color=ACCENT
                ))
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=segment_values,
                    theta=categories,
                    fill='toself',
                    name='Segment Average',
                    line_color=SECONDARY,
                    opacity=0.6
                ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 1])
                    ),
                    showlegend=True,
                    title="Design Profile vs Market Segment",
                    height=400,
                    paper_bgcolor=CHART_BG
                )
                
                st.plotly_chart(fig_radar, use_container_width=True)
            
            with comp_col2:
                # Success trajectory of similar games
                fig_trajectory = go.Figure()
                
                for _, game in neighbors.head(5).iterrows():
                    fig_trajectory.add_trace(go.Scatter(
                        x=[game["GameWeight"]],
                        y=[game["AvgRating"]],
                        mode='markers+text',
                        text=[game["Name"][:15]],
                        textposition="top center",
                        marker=dict(size=game["Owned Users"]/1000, color=game["Year Published"]),
                        showlegend=False,
                        hovertemplate=f"<b>{game['Name']}</b><br>Rating: {game['AvgRating']:.2f}<br>Owners: {game['Owned Users']:,}<extra></extra>"
                    ))
                
                # Add your game prediction
                fig_trajectory.add_trace(go.Scatter(
                    x=[complexity],
                    y=[predicted_rating],
                    mode='markers',
                    name='Your Game (Predicted)',
                    marker=dict(size=20, color='red', symbol='star'),
                    hovertemplate=f"<b>Your Design</b><br>Predicted Rating: {predicted_rating:.2f}<extra></extra>"
                ))
                
                fig_trajectory.update_layout(
                    title="Similar Games Performance Map",
                    xaxis_title="Complexity",
                    yaxis_title="Rating",
                    height=400,
                    paper_bgcolor=CHART_BG,
                    showlegend=True
                )
                
                st.plotly_chart(fig_trajectory, use_container_width=True)
            
            # Export options
            st.markdown("### 💾 Export Your Analysis")
            
            export_cols = st.columns(3)
            
            with export_cols[0]:
                analysis_data = {
                    "Design": [current_preset],
                    "Predicted_Rating": [predicted_rating],
                    "Expected_Owners": [predicted_owners],
                    "Success_Probability": [confidence],
                    "Complexity": [complexity],
                    "Play_Time": [play_time],
                    "Target_Price": [target_price]
                }
                analysis_df = pd.DataFrame(analysis_data)
                
                st.download_button(
                    "📊 Download Analysis Report",
                    data=analysis_df.to_csv(index=False).encode('utf-8'),
                    file_name=f"game_analysis_{current_preset.replace(' ', '_').lower()}.csv",
                    mime="text/csv"
                )
            
            with export_cols[1]:
                st.download_button(
                    "🎮 Download Similar Games",
                    data=neighbor_display.to_csv(index=False).encode('utf-8'),
                    file_name=f"similar_games_{current_preset.replace(' ', '_').lower()}.csv",
                    mime="text/csv"
                )
            
            with export_cols[2]:
                # Generate a simple text report
                report = f"""
BOARD GAME DESIGN ANALYSIS REPORT
==================================
Design Archetype: {current_preset}
Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}

PREDICTED PERFORMANCE
--------------------
Expected Rating: {predicted_rating:.2f}/10
Expected Owners: {predicted_owners:,}
Success Probability: {confidence}%
Market Segment: {market_size}

KEY SPECIFICATIONS
-----------------
Complexity: {complexity}/5.0
Play Time: {play_time} minutes
Player Count: {min_players}-{max_players}
Minimum Age: {min_age}+
Target Price: ${target_price}

TOP RECOMMENDATIONS
------------------
{chr(10).join([f'{i+1}. {rec}' for i, rec in enumerate(recommendations[:3])])}

SIMILAR SUCCESSFUL GAMES
-----------------------
{chr(10).join([f"- {row['Name']} (Rating: {row['AvgRating']:.2f}, Owners: {row['Owned Users']:,})" 
               for _, row in neighbors.head(3).iterrows()])}
                """
                
                st.download_button(
                    "📄 Download Full Report",
                    data=report.encode('utf-8'),
                    file_name=f"design_report_{current_preset.replace(' ', '_').lower()}.txt",
                    mime="text/plain"
                )

# Trend Analysis Tab
with tab_trends:
    st.markdown("## 📈 Market Trend Analysis & Forecasting")
    
    # Trend timeline selector
    trend_years = st.slider(
        "Select Analysis Period",
        int(view_f["Year Published"].min()),
        int(view_f["Year Published"].max()),
        (2010, CURRENT_YEAR),
        key="trend_years"
    )
    
    trend_data = view_f[(view_f["Year Published"] >= trend_years[0]) & 
                        (view_f["Year Published"] <= trend_years[1])]
    
    # Rising and falling mechanics
    st.markdown("### 🔄 Mechanic Trends: Rising & Falling")
    
    mech_trend_cols = st.columns(2)
    
    with mech_trend_cols[0]:
        st.markdown("#### 📈 Rising Mechanics")
        
        # Calculate mechanic trends
        recent_period = trend_data[trend_data["Year Published"] >= trend_years[1] - 2]
        older_period = trend_data[(trend_data["Year Published"] >= trend_years[0]) & 
                                  (trend_data["Year Published"] < trend_years[0] + 3)]
        
        rising_mechanics = []
        for mech in [c for c in trend_data.columns if c.startswith("Mechanic_")][:50]:
            recent_usage = recent_period[mech].mean() if len(recent_period) > 0 else 0
            older_usage = older_period[mech].mean() if len(older_period) > 0 else 0
            
            if recent_usage > older_usage * 1.5 and recent_usage > 0.05:
                growth = (recent_usage - older_usage) / max(0.01, older_usage) * 100
                rising_mechanics.append({
                    "Mechanic": mech.replace("Mechanic_", "").replace("_", " "),
                    "Growth": growth,
                    "Current Usage": recent_usage * 100,
                    "Quality": recent_period[recent_period[mech] == 1]["AvgRating"].mean() if len(recent_period[recent_period[mech] == 1]) > 0 else 0
                })
        
        if rising_mechanics:
            rising_df = pd.DataFrame(rising_mechanics).sort_values("Growth", ascending=False).head(10)
            
            for _, row in rising_df.iterrows():
                st.markdown(f"""
                <div style='margin: 0.5rem 0; padding: 0.5rem; background: {CHART_BG}; border-radius: 4px;'>
                    <strong>{row['Mechanic']}</strong><br>
                    <span style='color: {SUCCESS_COLOR}'>↑ {row['Growth']:.0f}% growth</span> | 
                    Usage: {row['Current Usage']:.1f}% | 
                    Avg Rating: {row['Quality']:.2f}
                </div>
                """, unsafe_allow_html=True)
    
    with mech_trend_cols[1]:
        st.markdown("#### 📉 Declining Mechanics")
        
        declining_mechanics = []
        for mech in [c for c in trend_data.columns if c.startswith("Mechanic_")][:50]:
            recent_usage = recent_period[mech].mean() if len(recent_period) > 0 else 0
            older_usage = older_period[mech].mean() if len(older_period) > 0 else 0
            
            if recent_usage < older_usage * 0.7 and older_usage > 0.05:
                decline = (older_usage - recent_usage) / older_usage * 100
                declining_mechanics.append({
                    "Mechanic": mech.replace("Mechanic_", "").replace("_", " "),
                    "Decline": decline,
                    "Current Usage": recent_usage * 100,
                    "Peak Usage": older_usage * 100
                })
        
        if declining_mechanics:
            declining_df = pd.DataFrame(declining_mechanics).sort_values("Decline", ascending=False).head(10)
            
            for _, row in declining_df.iterrows():
                st.markdown(f"""
                <div style='margin: 0.5rem 0; padding: 0.5rem; background: {CHART_BG}; border-radius: 4px;'>
                    <strong>{row['Mechanic']}</strong><br>
                    <span style='color: {DANGER_COLOR}'>↓ {row['Decline']:.0f}% decline</span> | 
                    Current: {row['Current Usage']:.1f}% | 
                    Peak: {row['Peak Usage']:.1f}%
                </div>
                """, unsafe_allow_html=True)
    
    # Theme evolution
    st.markdown("### 🎨 Theme Evolution Over Time")
    
    theme_evolution = {}
    years = sorted(trend_data["Year Published"].unique())
    
    for theme in [c for c in trend_data.columns if c.startswith("Cat:")][:10]:
        theme_evolution[theme] = []
        for year in years:
            year_data = trend_data[trend_data["Year Published"] == year]
            usage = year_data[theme].mean() if len(year_data) > 0 else 0
            theme_evolution[theme].append(usage)
    
    fig_themes = go.Figure()
    
    for theme, values in theme_evolution.items():
        if max(values) > 0.05:  # Only show themes with >5% usage at some point
            fig_themes.add_trace(go.Scatter(
                x=years,
                y=[v * 100 for v in values],
                mode='lines',
                name=theme.replace("Cat:", "").replace("_", " "),
                line=dict(width=2)
            ))
    
    fig_themes.update_layout(
        title="Theme Popularity Evolution",
        xaxis_title="Year",
        yaxis_title="Usage %",
        height=400,
        paper_bgcolor=CHART_BG,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_themes, use_container_width=True)

    narr("""
    **Where the puck is going.** Themes cycle, but the constant is respect for time. 
    Designers are packing strategy into tighter sessions. That is not dumbing things down. 
    That is improving game design and smarter players. If you are designing into the near future, pair a strong theme with clean 
    teach, depth that reveals across plays, and a clear promise on the box about time.
    """)

    # Complexity creep analysis
    st.markdown("### 🧩 The Complexity Creep Phenomenon")
    
    complexity_by_year = trend_data.groupby("Year Published").agg({
        "GameWeight": ["mean", "median", "std"],
        "AvgRating": "mean",
        "Name": "count"
    }).reset_index()
    
    complexity_by_year.columns = ["Year", "Mean_Complexity", "Median_Complexity", 
                                  "Std_Complexity", "Avg_Rating", "Game_Count"]
    
    fig_complexity = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Complexity Evolution", "Complexity vs Success")
    )
    
    fig_complexity.add_trace(
        go.Scatter(x=complexity_by_year["Year"], 
                  y=complexity_by_year["Mean_Complexity"],
                  mode='lines+markers',
                  name='Mean Complexity',
                  line=dict(color=CHART_COLORS[0], width=3)),
        row=1, col=1
    )
    
    fig_complexity.add_trace(
        go.Scatter(x=complexity_by_year["Year"],
                  y=complexity_by_year["Median_Complexity"],
                  mode='lines',
                  name='Median Complexity',
                  line=dict(color=CHART_COLORS[1], width=2, dash='dash')),
        row=1, col=1
    )
    
    fig_complexity.add_trace(
        go.Scatter(x=complexity_by_year["Mean_Complexity"],
                  y=complexity_by_year["Avg_Rating"],
                  mode='markers',
                  marker=dict(size=complexity_by_year["Game_Count"]/10, 
                             color=complexity_by_year["Year"],
                             colorscale='Viridis'),
                  text=complexity_by_year["Year"],
                  hovertemplate="Year: %{text}<br>Complexity: %{x:.2f}<br>Avg Rating: %{y:.2f}<extra></extra>",
                  showlegend=False),
        row=1, col=2
    )
    
    fig_complexity.update_xaxes(title_text="Year", row=1, col=1)
    fig_complexity.update_xaxes(title_text="Average Complexity", row=1, col=2)
    fig_complexity.update_yaxes(title_text="Complexity (1-5)", row=1, col=1)
    fig_complexity.update_yaxes(title_text="Average Rating", row=1, col=2)
    
    fig_complexity.update_layout(height=400, paper_bgcolor=CHART_BG)
    
    st.plotly_chart(fig_complexity, use_container_width=True)
    
    # Market predictions
    st.markdown("### 🔮 Market Predictions for Next 2 Years")
    
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    
    # Calculate trends for predictions
    recent_3_years = trend_data[trend_data["Year Published"] >= CURRENT_YEAR - 3]
    
    if len(recent_3_years) > 10:
        
        # Complexity predictions
        complexity_trend = trend_data.groupby("Year Published")["GameWeight"].mean()
        if len(complexity_trend) > 3:
            complexity_change = complexity_trend.iloc[-3:].diff().mean()
            predicted_complexity = complexity_trend.iloc[-1] + complexity_change
            
            direction = "increase" if complexity_change > 0 else "decrease"
            st.write(f"🧩 **Average complexity will {direction} to:** {predicted_complexity:.2f}/5.0")
        
        # Hot categories prediction
        st.write("🔥 **Predicted hot categories:**")
        
        hot_categories = []
        for cat in [c for c in recent_3_years.columns if c.startswith("Cat:")][:20]:
            recent_usage = recent_3_years[cat].mean()
            recent_rating = recent_3_years[recent_3_years[cat] == 1]["AvgRating"].mean()
            
            if recent_usage > 0.1 and recent_rating > 7.0:
                hot_categories.append((cat.replace("Cat:", ""), recent_rating))
        
        for cat, rating in sorted(hot_categories, key=lambda x: x[1], reverse=True)[:5]:
            st.write(f"  • {cat} (avg rating: {rating:.2f})")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Segment Explorer Tab
with tab_segments:
    st.markdown("## 🗺️ Market Segment Deep Dive")
    
    # Cluster selection
    cluster_options = sorted(view_f["Cluster"].unique())
    selected_cluster = st.selectbox(
        "Select Market Segment to Explore",
        cluster_options,
        format_func=lambda x: (
            f"{cluster_labels.get(x, f'Segment {x}')} "
            f"({cluster_insights[x]['size']} games, {cluster_insights[x]['avg_rating']:.2f} avg)"
        ),
    )
    cluster_data = view_f[view_f["Cluster"] == selected_cluster]
    
    # Segment overview
    st.markdown(f"### Segment {selected_cluster} Overview")
    
    overview_cols = st.columns(5)
    
    with overview_cols[0]:
        st.metric("Games", cluster_insights[selected_cluster]["size"])
        st.caption("Market size")
    
    with overview_cols[1]:
        st.metric("Avg Rating", f"{cluster_insights[selected_cluster]['avg_rating']:.2f}")
        st.caption("Quality indicator")
    
    with overview_cols[2]:
        st.metric("Complexity", f"{cluster_insights[selected_cluster]['avg_complexity']:.1f}")
        st.caption("Design sophistication")
    
    with overview_cols[3]:
        st.metric("Median Owners", f"{int(cluster_insights[selected_cluster]['avg_owners']):,}")
        st.caption("Market reach")
    
    with overview_cols[4]:
        opportunity = cluster_insights[selected_cluster]["opportunity_score"]
        color = SUCCESS_COLOR if opportunity > 70 else WARNING_COLOR if opportunity > 40 else DANGER_COLOR
        st.markdown(f"<h3 style='color: {color}'>{opportunity:.0f}%</h3>", unsafe_allow_html=True)
        st.caption("Opportunity score")
    
    narr("""
    **Reading a segment.** Look at three things. What the segment loves to do 
    (mechanics above thirty percent). How long the table sits (median playtime). 
    Where quality clusters (rating distribution). If your idea fights the segment norms, make the reason obvious and delightful. 
    These are not prognosi's for a game, but pitfalls to avoid in a cluster.
    """)
    # Segment characteristics
    segment_cols = st.columns(2)
    
    with segment_cols[0]:
        st.markdown("#### Defining Characteristics")
        
        # Find most common mechanics
        mech_usage = {}
        for mech in [c for c in cluster_data.columns if c.startswith("Mechanic_")][:50]:
            usage = cluster_data[mech].mean()
            if usage > 0.3:
                mech_usage[mech.replace("Mechanic_", "")] = usage
        
        if mech_usage:
            st.write("**Core Mechanics:**")
            for mech, usage in sorted(mech_usage.items(), key=lambda x: x[1], reverse=True)[:5]:
                st.write(f"• {mech}: {usage*100:.0f}% of games")
        
        # Find common themes
        theme_usage = {}
        for theme in [c for c in cluster_data.columns if c.startswith("Cat:")][:30]:
            usage = cluster_data[theme].mean()
            if usage > 0.3:
                theme_usage[theme.replace("Cat:", "")] = usage
        
        if theme_usage:
            st.write("**Dominant Themes:**")
            for theme, usage in sorted(theme_usage.items(), key=lambda x: x[1], reverse=True)[:5]:
                st.write(f"• {theme}: {usage*100:.0f}% of games")
    
    with segment_cols[1]:
        st.markdown("#### 📊 Performance Distribution")
        
        fig_dist = go.Figure()
        
        fig_dist.add_trace(go.Histogram(
            x=cluster_data["AvgRating"],
            nbinsx=20,
            name="Rating Distribution",
            marker_color=CHART_COLORS[0],
            opacity=0.7
        ))
        
        fig_dist.add_vline(x=cluster_data["AvgRating"].median(), 
                          line_dash="dash", 
                          line_color="red",
                          annotation_text="Median")
        
        fig_dist.update_layout(
            title="Rating Distribution in Segment",
            xaxis_title="Average Rating",
            yaxis_title="Number of Games",
            height=300,
            paper_bgcolor=CHART_BG,
            showlegend=False
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # Top performers
    st.markdown("#### 🏆 Top 10 Games in This Segment")
    
    top_segment_games = cluster_data.nlargest(10, "AvgRating")[
        ["Name", "Year Published", "AvgRating", "Owned Users", "GameWeight", "Play Time"]
    ]
    
    st.dataframe(
        top_segment_games.style.background_gradient(subset=["AvgRating"]),
        use_container_width=True,
        hide_index=True
    )
    
    # Segment evolution
    st.markdown("#### 📈 Segment Evolution Over Time")
    
    segment_yearly = cluster_data.groupby("Year Published").agg({
        "Name": "count",
        "AvgRating": "mean",
        "Owned Users": "median"
    }).reset_index()
    
    fig_segment_evolution = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Releases Per Year", "Quality Trend")
    )
    
    fig_segment_evolution.add_trace(
        go.Bar(x=segment_yearly["Year Published"],
               y=segment_yearly["Name"],
               marker_color=CHART_COLORS[0],
               name="Releases"),
        row=1, col=1
    )
    
    fig_segment_evolution.add_trace(
        go.Scatter(x=segment_yearly["Year Published"],
                  y=segment_yearly["AvgRating"],
                  mode='lines+markers',
                  line=dict(color=CHART_COLORS[1], width=2),
                  name="Avg Rating"),
        row=1, col=2
    )
    
    fig_segment_evolution.update_layout(
        height=300,
        paper_bgcolor=CHART_BG,
        showlegend=False
    )
    
    st.plotly_chart(fig_segment_evolution, use_container_width=True)
    narr("""
    **Entering the segment.** If releases are rising and ratings hold, the there is likely unmet demand. 
    If releases spike and ratings sag, thats a tough and saturated market. Add one new idea that matters in a high demand space.
    """)
    # Opportunities in segment
    st.markdown("#### 💡 Opportunities in This Segment")
    
    segment_recommendations = generate_design_recommendations(cluster_data, view_f)
    
    st.markdown('<div class="recommendation-card">', unsafe_allow_html=True)
    for rec in segment_recommendations:
        st.write(rec)
    st.markdown('</div>', unsafe_allow_html=True)

# Mechanic Synergies Tab
with tab_synergies:
    st.markdown("## 🔗 Mechanic Synergy Analysis")
    st.markdown("*Discover powerful mechanic combinations that drive success*")
    
    # Calculate synergies
    synergies = identify_mechanic_synergies(view_f, min_games=20)
    
    if len(synergies) > 0:
        # Top synergies table
        st.markdown("### 🏆 Top Mechanic Combinations")
        
        top_synergies = synergies.head(20)
        
        # Enhanced display with success indicators
        for i, row in top_synergies.iterrows():
            if i < 10:  # Show top 10 in detail
                success_color = SUCCESS_COLOR if row["Success Rate"] > 0.3 else WARNING_COLOR if row["Success Rate"] > 0.15 else DANGER_COLOR
                
                st.markdown(f"""
                <div style='margin: 1rem 0; padding: 1rem; background: {CHART_BG}; border-left: 4px solid {success_color}; border-radius: 4px;'>
                    <h4>{row['Mechanic 1']} + {row['Mechanic 2']}</h4>
                    <div style='display: flex; justify-content: space-between;'>
                        <span>📊 {row['Games']} games</span>
                        <span>⭐ {row['Avg Rating']:.2f} avg rating</span>
                        <span>🎯 {row['Success Rate']*100:.1f}% success rate</span>
                        <span>👥 {int(row['Avg Owners']):,} median owners</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Network visualization
        # --- Interactive network ---
        st.markdown("### 🌐 Mechanic Relationship Network")
        fig_network = build_mech_network_fig_static(synergies, top_n=30)
        st.plotly_chart(fig_network, use_container_width=True)
        
        narr("""
        **Mechanics that work.** Certain combinations produce high ratings. Resist feature creep. Anchor on one or two core systems 
        and let the rest emerge through playtesting.
        """)
        
        # Underexplored combinations
        st.markdown("### 💎 Underexplored High-Potential Combinations")
        
        underexplored = synergies[(synergies["Games"] < 50) & (synergies["Success Rate"] > 0.25)]
        
        if len(underexplored) > 0:
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            st.write("**These combinations show high success rates but are rarely used:**")
            
            for _, row in underexplored.head(5).iterrows():
                st.write(f"• **{row['Mechanic 1']} + {row['Mechanic 2']}**: Only {row['Games']} games, but {row['Success Rate']*100:.0f}% success rate")
            
            st.markdown('</div>', unsafe_allow_html=True)
        narr("""
        **High upside, low volume.** Underplayed combinations with strong success rates are design invitations. Prototype fast. Play with real people. If the table smiles without your help, you are close.
        """)
        
        # Mechanic pairing recommendations by complexity
        st.markdown("### 🎯 Recommended Pairings by Complexity Level")
        
        complexity_levels = [
            ("Simple Games (1.0-2.0)", 1.0, 2.0),
            ("Medium Games (2.0-3.0)", 2.0, 3.0),
            ("Complex Games (3.0-5.0)", 3.0, 5.0)
        ]
        
        rec_cols = st.columns(3)
        
        for i, (level_name, min_c, max_c) in enumerate(complexity_levels):
            with rec_cols[i]:
                st.markdown(f"#### {level_name}")
                
                level_games = view_f[(view_f["GameWeight"] >= min_c) & (view_f["GameWeight"] < max_c)]
                level_synergies = identify_mechanic_synergies(level_games, min_games=10)
                
                if len(level_synergies) > 0:
                    for _, row in level_synergies.head(3).iterrows():
                        st.write(f"• {row['Mechanic 1']} + {row['Mechanic 2']}")
                        st.caption(f"  {row['Success Rate']*100:.0f}% success")
    else:
        st.info("Not enough data to calculate mechanic synergies with current filters")

# Footer 
narr("""
**Bottom line.** Games do not suck anymore. 
The average modern title beats the best games of the 90's that started the boom. The reason is simple. 
Designers learned to respect time, balance rules, create novel mechanics, and make the first play feel good. 
You have to find a demand and then follow that model.
""")
st.markdown("---")




















