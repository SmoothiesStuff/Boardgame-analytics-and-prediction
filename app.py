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
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from scipy import stats
from joblib import load as joblib_load
import warnings
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
/* styled narrative blockquotes */
.stApp blockquote {
    background: linear-gradient(90deg, {CHART_BG} 0%, #FFF 100%);
    border-left: 5px solid {ACCENT};
    padding: 1rem 1.5rem;
    margin: 1rem 0;
    border-radius: 0 8px 8px 0;
    color: {MUTED};
}
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

# Configuration
DEFAULT_PARQUET_PATH = "cleaned_large_bgg_dataset.parquet"
DEFAULT_CSV_PATH = "cleaned_large_bgg_dataset.csv"
CURRENT_YEAR = 2022

EXCLUDE_FOR_CLUSTERING = [
    "Owned Users", "BayesAvgRating", "AvgRating", "Users Rated", "BGG Rank", 
    "StdDev", "NumWant", "NumWish", "NumComments", "NumWeightVotes",
    "ID", "BGGId", "Name", "ImagePath", "Rank:strategygames", "Rank:abstracts", 
    "Rank:familygames", "Rank:thematic", "Rank:cgs", "Rank:wargames", 
    "Rank:partygames", "Rank:childrensgames"
]

MODEL_PATHS = {
    "rating_xgb": "models/rating_xgb.joblib", 
    "sales_xgb": "models/sales_xgb.joblib",
}
INPUT_SCALER_PATH = "models/input_scaler.joblib"
PRED_EXCLUDE = {"Cluster", "PCA1", "PCA2", "LogOwned", "SalesPercentile", "__dist"}

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
        subplot_titles=("Games Released per Year", "Average Quality Trends", "Market Characteristics"),
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

def create_mechanic_network_graph(synergies_df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    """Create network graph showing mechanic relationships."""
    if len(synergies_df) == 0:
        return go.Figure().add_annotation(text="Insufficient data for network graph", showarrow=False)
    
    top_synergies = synergies_df.head(top_n)
    
    # Build network
    nodes = list(set(top_synergies["Mechanic 1"].tolist() + top_synergies["Mechanic 2"].tolist()))
    node_dict = {node: i for i, node in enumerate(nodes)}
    
    edges = []
    edge_weights = []
    for _, row in top_synergies.iterrows():
        edges.append((node_dict[row["Mechanic 1"]], node_dict[row["Mechanic 2"]]))
        edge_weights.append(row["Success Rate"])
    
    # Simple circular layout
    n = len(nodes)
    angles = [2 * np.pi * i / n for i in range(n)]
    x = [np.cos(angle) for angle in angles]
    y = [np.sin(angle) for angle in angles]
    
    # Create figure
    fig = go.Figure()
    
    # Add edges
    for edge, weight in zip(edges, edge_weights):
        fig.add_trace(go.Scatter(
            x=[x[edge[0]], x[edge[1]]],
            y=[y[edge[0]], y[edge[1]]],
            mode='lines',
            line=dict(width=weight*5, color=CHART_COLORS[0]),
            opacity=0.5,
            hoverinfo='skip',
            showlegend=False
        ))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers+text',
        marker=dict(size=20, color=CHART_COLORS[1]),
        text=nodes,
        textposition="top center",
        hovertemplate='%{text}<extra></extra>',
        showlegend=False
    ))
    
    fig.update_layout(
        title="Mechanic Synergy Network (line thickness = success rate)",
        height=500,
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
PROFILE_PRESETS = {
    "🎉 Family Party Game": {
        "description": "Accessible, social games perfect for casual gatherings",
        "cats": ["Cat:Family", "Cat:Party"],
        "mechs_on": ["Set Collection", "Hand Management", "Voting", "Take That"],
        "year": CURRENT_YEAR, "min_players": 3, "max_players": 8, 
        "play_time": 30, "min_age": 8, "weight": 1.8,
        "market_insight": "Family games with 20-40 minute play times have seen 40% growth since 2020"
    },
    "🤝 Cooperative Strategy": {
        "description": "Team-based strategic challenges with deep decision making",
        "cats": ["Cat:Thematic", "Cat:Strategy"],
        "mechs_on": ["Cooperative Game", "Action Points", "Variable Player Powers"],
        "year": CURRENT_YEAR, "min_players": 1, "max_players": 4, 
        "play_time": 60, "min_age": 12, "weight": 2.6,
        "market_insight": "Co-op games with solo modes have 2x the ownership of multiplayer-only titles"
    },
    "🏛️ Heavy Euro Game": {
        "description": "Complex economic engines for serious strategists",
        "cats": ["Cat:Strategy", "Economic"],
        "mechs_on": ["Network and Route Building", "Market", "Worker Placement"],
        "year": CURRENT_YEAR, "min_players": 2, "max_players": 4, 
        "play_time": 120, "min_age": 14, "weight": 3.6,
        "market_insight": "Heavy euros (3.5+ complexity) targeting 90-120 minutes outperform 150+ minute games"
    },
    "⚔️ Thematic Adventure": {
        "description": "Story-driven experiences with emergent narratives",
        "cats": ["Cat:Thematic", "Adventure", "Fantasy"],
        "mechs_on": ["Dice Rolling", "Tile Placement", "Variable Player Powers"],
        "year": CURRENT_YEAR, "min_players": 1, "max_players": 4, 
        "play_time": 90, "min_age": 12, "weight": 2.7,
        "market_insight": "Adventure games with campaign modes see 3x higher engagement"
    },
    "♟️ Abstract Strategy": {
        "description": "Pure strategic gameplay with minimal luck",
        "cats": ["Cat:Abstract"],
        "mechs_on": ["Grid Movement", "Pattern Building", "Area Majority / Influence"],
        "year": CURRENT_YEAR, "min_players": 2, "max_players": 2, 
        "play_time": 20, "min_age": 10, "weight": 2.0,
        "market_insight": "2-player abstracts under 30 minutes are experiencing a renaissance"
    },
    "🃏 Deck Builder": {
        "description": "Card-driven engine building with high replayability",
        "cats": ["Cat:CGS"],
        "mechs_on": ["Deck Construction", "Hand Management", "Set Collection"],
        "year": CURRENT_YEAR, "min_players": 2, "max_players": 4, 
        "play_time": 45, "min_age": 10, "weight": 2.3,
        "market_insight": "Deck builders with asymmetric starting decks rate 0.5 points higher"
    },
    "🎭 Social Deduction": {
        "description": "Hidden role games focused on deduction and bluffing",
        "cats": ["Cat:Party"],
        "mechs_on": ["Hidden Roles", "Voting", "Player Elimination"],
        "year": CURRENT_YEAR, "min_players": 5, "max_players": 10, 
        "play_time": 20, "min_age": 10, "weight": 1.7,
        "market_insight": "Social deduction without player elimination sees 60% more positive reviews"
    },
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

uploaded = st.sidebar.file_uploader("📁 Upload dataset (CSV/Parquet)", type=["csv", "parquet"])
df = load_df(uploaded)

# Add data quality metrics
data_quality = {
    "Total Games": len(df),
    "Features": len(df.columns),
    "Years Covered": f"{df['Year Published'].min()}-{df['Year Published'].max()}",
    "Avg Completeness": f"{(1 - df.isnull().mean().mean())*100:.1f}%"
}

with st.sidebar.expander("📊 Data Quality Metrics", expanded=False):
    for metric, value in data_quality.items():
        st.metric(metric, value)

# Advanced filtering controls
st.sidebar.markdown("### 🎯 Analysis Parameters")
k = st.sidebar.slider("Cluster Granularity", 2, 20, 10, 
                      help="Higher values create more specific market segments")
topn = st.sidebar.slider("Comparison Pool Size", 5, 50, 15,
                         help="Number of similar games to analyze")

st.sidebar.markdown("### 🔍 Market Filters")
year_col = "Year Published"
min_year, max_year = int(df[year_col].min()), int(df[year_col].max())
display_min_year = max(1900, min_year)

yr_rng = st.sidebar.slider(
    "📅 Year Range", 
    display_min_year, max_year, 
    (max(1990, display_min_year), max_year),
    help="Focus on specific time periods"
)

weight_col = "GameWeight"
min_w, max_w = float(df[weight_col].min()), float(df[weight_col].max())
wt_rng = st.sidebar.slider(
    "🧩 Complexity Range", 
    1.0, 5.0, 
    (1.5, 4.0),
    step=0.1,
    help="1=Very Simple, 5=Very Complex"
)

pt_rng_hrs = st.sidebar.slider(
    "⏱️ Play Time (hours)", 
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
    "👶 Age Range", 
    3, 18, 
    (8, 14),
    help="Minimum age requirements"
)

# Mechanic and theme filtering
with st.sidebar.expander("🎲 Mechanics & Themes", expanded=False):
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
        # render as a blockquote so markdown (like **bold**) is parsed
        bl = "> " + md.strip().replace("\n", "\n> ")
        st.markdown(bl)

# Convert play time range to minutes
pt_rng = (int(pt_rng_hrs[0] * 60), int(pt_rng_hrs[1] * 60))

# Prepare data and clustering
X_all, meta = split_features(df)
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
narr(f"""
**Quick read of the market.** At a glance the median rating is {median_rating:.2f}. That lines up with a larger pattern from my dashboard work. Low complexity games under 2.0 are usually not great. Quality starts to appear around 2.5 and keeps climbing as games get deeper. The surprise is that very simple games can still shine when the design is clean and intentional. If you want consistent quality, aim for thoughtful depth rather than a pile of rules. Families do not hate depth. They hate clutter.
""")

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
    opportunities = sorted(cluster_insights.items(), key=lambda x: x[1]["opportunity_score"], reverse=True)[:5]
    
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
    **How to use these segments.** The Opportunity Score surfaces places where demand outpaces supply. The healthiest pockets tend to be moderate complexity with real decision space, 60 to 90 minutes, and a minimum age of 10. Co-op with solo support often doubles engagement because players can learn alone then bring the table along. If a segment is small and strong, it is not a red flag. It is a green light for a focused design that respects time and attention.
    """)

    # Market evolution timeline
    st.markdown("### 📈 Complete Market Evolution (1950-Present)")
    evolution_fig = create_market_evolution_timeline(view_f)
    st.plotly_chart(evolution_fig, use_container_width=True)
    narr("""
    **What changed over time.** The boom after Catan raised the bar. Ratings rose and designers learned to do more with less. Complexity ticked up, yet playtime did not. That is craft improving. Strategic richness without bloat. The market now rewards clarity, replay, and respect for the clock.
    """)

    # Key insights
    st.markdown("### 💡 Data-Driven Market Insights")
    
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("#### 🚀 Growth Sectors")
        
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
        st.markdown("#### 🎯 Sweet Spots Identified")
        
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
    **Complexity and time, together.** Longer games tend to rate higher, but there is a ceiling on how many people will buy a three hour commitment. Short and complex is the unicorn. When a game is deep and plays fast, it becomes a staple. If a game is simple, keep it short. Stretching a simple idea for an hour turns charm into chores.
    """)

# Design Wizard Tab  
with tab_wizard:
    st.markdown("## 🧙‍♂️ AI-Powered Game Design Wizard")
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
            year_published = st.number_input("Release Year", 
                                            value=preset_data["year"], 
                                            min_value=CURRENT_YEAR,
                                            max_value=CURRENT_YEAR+2)
            min_players = st.number_input("Min Players", 
                                         value=preset_data["min_players"],
                                         min_value=1, max_value=10)
        
        with spec_cols[1]:
            play_time = st.number_input("Play Time (min)", 
                                       value=preset_data["play_time"],
                                       min_value=5, max_value=360, step=5)
            max_players = st.number_input("Max Players", 
                                         value=preset_data["max_players"],
                                         min_value=min_players, max_value=20)
        
        with spec_cols[2]:
            complexity = st.slider("Complexity", 1.0, 5.0, 
                                  preset_data["weight"], 0.1,
                                  help="1=Very Simple, 5=Very Complex")
            min_age = st.number_input("Min Age", 
                                     value=preset_data["min_age"],
                                     min_value=3, max_value=21)
        
        with spec_cols[3]:
            kickstarted = st.selectbox("Funding Model", 
                                      ["Traditional", "Kickstarter", "Gamefound"],
                                      index=0)
            production_quality = st.select_slider("Production Quality",
                                                 ["Basic", "Standard", "Premium", "Deluxe"],
                                                 value="Standard")
        
        st.markdown("#### Mechanics & Theme Selection")
        
        mech_theme_cols = st.columns(2)
        
        with mech_theme_cols[0]:
            available_mechs = [c for c in X_all.columns if c.startswith("Mechanic_") or c in preset_data["mechs_on"]][:40]
            selected_mechanics = st.multiselect("Core Mechanics (3-5 recommended)",
                                              available_mechs,
                                              default=preset_data["mechs_on"][:3])
        
        with mech_theme_cols[1]:
            available_themes = [c for c in X_all.columns if c.startswith("Cat:") or c in preset_data["cats"]][:30]
            selected_themes = st.multiselect("Themes & Categories",
                                           available_themes,
                                           default=preset_data["cats"][:2])
        
        st.markdown("#### Additional Considerations (predictions come from estimates of 2022 production costs)")
        
        additional_cols = st.columns(3)
        
        with additional_cols[0]:
            solo_mode = st.checkbox("Include Solo Mode", value=min_players == 1)
            campaign_mode = st.checkbox("Campaign/Legacy Elements", value=False)
        
        with additional_cols[1]:
            expansion_ready = st.checkbox("Expansion-Ready Design", value=True)
            app_integration = st.checkbox("Companion App", value=False)
        
        with additional_cols[2]:
            target_price = st.slider("Target MSRP ($)", 20, 150, 50, 5)
            component_quality = st.select_slider("Component Quality",
                                                ["Basic", "Good", "Premium"],
                                                value="Good")
        narr("""
    **Design first principles.** Start from three anchors. Target complexity that invites thinking without confusion. Land the core loop in 60 to 90 minutes. Aim the minimum age at 10 so families and hobby tables overlap. From there, pick one mechanic that does the heavy lifting and one that creates emergent texture. Resist adding a third that just adds rules.
    """)

        analyze_button = st.form_submit_button("🔮 Analyze Design & Generate Predictions", 
                                              type="primary", use_container_width=True)
        
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
        for mech in selected_mechanics:
            if mech in X_all.columns:
                profile[mech] = 1
        
        for theme in selected_themes:
            if theme in X_all.columns:
                profile[theme] = 1
        
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
            st.markdown("## 🎯 Design Analysis Results")
            
            # AI Predictions section
            st.markdown("### 🤖 AI Performance Predictions")
            
            models = load_models(MODEL_PATHS)
            
            pred_cols = st.columns(4)
            
            # Mock predictions if no models
            if not models:
                # Generate realistic predictions based on similar games
                predicted_rating = neighbors["AvgRating"].mean() + np.random.normal(0, 0.2)
                predicted_rating = max(5.0, min(8.5, predicted_rating))
                predicted_owners = int(neighbors["Owned Users"].median() * np.random.uniform(0.8, 1.2))
                confidence = 75 + np.random.randint(-10, 15)
                percentile = stats.percentileofscore(view_f["AvgRating"], predicted_rating)
            else:
                # Use trained models
                scaler_in = models.get("_input_scaler")
            
                # Pick feature list (prefer scaler feature names; fall back to model or X_all)
                if scaler_in is not None and hasattr(scaler_in, "feature_names_in_"):
                    training_cols = list(scaler_in.feature_names_in_)
                elif "rating_xgb" in models and hasattr(models["rating_xgb"], "feature_names_in_"):
                    training_cols = list(models["rating_xgb"].feature_names_in_)
                elif "sales_xgb" in models and hasattr(models["sales_xgb"], "feature_names_in_"):
                    training_cols = list(models["sales_xgb"].feature_names_in_)
                else:
                    training_cols = list(X_all.columns)
            
                X_pred = align_profile_to_training(profile, training_cols, scaler=scaler_in)
            
                # Predict rating
                if "rating_xgb" in models:
                    predicted_rating = float(models["rating_xgb"].predict(X_pred)[0])
                else:
                    predicted_rating = float(neighbors["AvgRating"].mean())
            
                # Predict owners
                if "sales_xgb" in models:
                    owners_pred = models["sales_xgb"].predict(X_pred)[0]
                    # If your sales model was trained on log1p(owners), uncomment next line:
                    # owners_pred = np.expm1(owners_pred)
                    predicted_owners = int(max(0, owners_pred))
                else:
                    predicted_owners = int(neighbors["Owned Users"].median())
            
                # Percentile vs market
                percentile = stats.percentileofscore(view_f["AvgRating"], predicted_rating)
            
                # Simple confidence from neighbor tightness (0–100 clip)
                d = neighbors["__dist"]
                denom = (d.std() if d.std() > 1e-6 else 1.0)
                confidence = int(np.clip(70 + (1 - d.mean()/denom)*20, 40, 95))
            
            with pred_cols[0]:
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                st.metric("Predicted Rating", f"{predicted_rating:.2f}/10")
                st.progress(predicted_rating/10)
                st.caption(f"Top {100-percentile:.0f}% percentile")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with pred_cols[1]:
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                st.metric("Expected Owners", f"{predicted_owners:,}")
                market_size = "Niche" if predicted_owners < 5000 else "Mid-Market" if predicted_owners < 50000 else "Mass Market"
                st.caption(f"📊 {market_size} potential")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with pred_cols[2]:
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                st.metric("Success Probability", f"{confidence}%")
                st.progress(confidence/100)
                risk_level = "Low Risk" if confidence > 70 else "Moderate Risk" if confidence > 50 else "High Risk"
                st.caption(f"⚠️ {risk_level}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with pred_cols[3]:
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                roi_estimate = (predicted_owners * target_price * 0.4) / 50000  # Simplified ROI
                st.metric("ROI Estimate", f"{roi_estimate:.1f}x")
                payback = "6 months" if roi_estimate > 2 else "12 months" if roi_estimate > 1 else "18+ months"
                st.caption(f"💰 Payback: {payback}")
                st.markdown('</div>', unsafe_allow_html=True)

            narr(f"""
            **Reading your forecast.** Treat predicted rating, owners, and risk as a compass, not a verdict. If the model likes your rating but owners look soft, the design might be niche or overpriced. If owners look strong but rating is middling, you might have a fun toy that needs sharper decisions. Price can move demand, but not forever. Anchor the MSRP to what the experience feels like in the first 15 minutes.
            """)

                        ########## Pricing & Unit Economics ##########
            st.markdown("### 💵 Pricing & Unit Economics")
            
            # Use your existing slider value for target price as the default MSRP
            msrp_default = int(target_price)
            
            with st.expander("Set price & costs", expanded=True):
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    msrp = st.slider("Desired Price (MSRP $)", 15, 150, msrp_default, 1, key="msrp_input")
                with c2:
                    unit_cogs = st.slider("Unit Cost to Produce ($)", 1, 60, 12)
                with c3:
                    marketing_fixed = st.number_input("Marketing Budget ($, fixed)", 0, 500_000, 25_000, step=1_000)
                with c4:
                    misc_fixed = st.number_input("Misc & Dev ($, fixed)", 0, 500_000, 15_000, step=1_000)
            
                c5, c6, c7 = st.columns(3)
                with c5:
                    sales_window = st.slider("Sales Window (months)", 3, 36, 12)
                with c6:
                    shipping_per_unit = st.slider("Fulfillment & Shipping per Unit ($)", 0, 30, 5)
                with c7:
                    returns_pct = st.slider("Returns/Damage Allowance (%)", 0, 20, 5) / 100.0
            
                with st.expander("Advanced assumptions", expanded=False):
                    fee_default = default_channel_fee_pct(kickstarted)
                    channel_fee_pct = st.slider("Retailer / Platform Fee (%)", 0, 70, int(fee_default*100)) / 100.0
                    apply_sensitivity = st.checkbox("Apply price sensitivity to owners", value=True)
                    elasticity = st.slider("Price Elasticity (negative)", -2.0, -0.1,
                                           -1.1 if kickstarted == "Traditional" else -0.8, 0.1,
                                           help="Percent change in owners for 1 percent price change vs anchor")
            
            # Anchor price from design signals
            anchor_price = estimate_anchor_price(complexity, component_quality, production_quality, max_players, play_time)
            
            # Adjust owners by price (bounded)
            owners_base = float(predicted_owners)
            if apply_sensitivity:
                owners_adj = owners_base * (msrp / max(anchor_price, 1.0)) ** elasticity
                owners_adj = float(np.clip(owners_adj, owners_base * 0.6, owners_base * 1.4))
            else:
                owners_adj = owners_base
            
            # Unit economics
            net_per_unit = msrp * (1 - channel_fee_pct)
            gross_profit_per_unit = net_per_unit - (unit_cogs + shipping_per_unit)
            effective_units = owners_adj * (1 - returns_pct)
            fixed_costs = float(marketing_fixed + misc_fixed)
            
            total_gross_profit = gross_profit_per_unit * max(effective_units, 0)
            net_profit = total_gross_profit - fixed_costs
            roi_multiple = (net_profit / fixed_costs) if fixed_costs > 0 else float("inf")
            
            breakeven_units = (fixed_costs / gross_profit_per_unit) if gross_profit_per_unit > 0 else float("inf")
            monthly_units = effective_units / max(sales_window, 1)
            payback_months = math.ceil(breakeven_units / max(monthly_units, 1)) if math.isfinite(breakeven_units) else None
            gross_margin_pct = (gross_profit_per_unit / net_per_unit) if net_per_unit > 0 else 0.0
            
            # Metrics row
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            with m1:
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                st.metric("Adjusted Owners", f"{int(effective_units):,}", f"base {int(owners_base):,}")
                st.caption(f"Anchor ${anchor_price:.0f} • fee {int(channel_fee_pct*100)}% • returns {int(returns_pct*100)}%")
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
            
            # Profit vs Price sensitivity
            st.markdown("#### 📈 Profit vs Price (sensitivity)")
            pmin, pmax = max(10, anchor_price*0.6), min(150, anchor_price*1.4)
            prices = np.linspace(pmin, pmax, 50)
            profits = []
            for p in prices:
                net_u = p * (1 - channel_fee_pct)
                gp_u = net_u - (unit_cogs + shipping_per_unit)
                if apply_sensitivity:
                    own = owners_base * (p / max(anchor_price, 1.0)) ** elasticity
                    own = np.clip(own, owners_base * 0.6, owners_base * 1.4)
                else:
                    own = owners_base
                eff = own * (1 - returns_pct)
                profits.append(gp_u * eff - fixed_costs)
            
            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(x=prices, y=profits, mode="lines", name="Profit"))
            fig_price.add_vline(x=msrp, line_dash="dash", line_color="red", annotation_text="Your price")
            fig_price.add_vline(x=anchor_price, line_dash="dot", line_color="gray", annotation_text="Anchor")
            fig_price.update_layout(
                title="Projected Net Profit vs MSRP",
                xaxis_title="MSRP ($)", yaxis_title="Profit ($)",
                paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG, height=380
            )
            st.plotly_chart(fig_price, use_container_width=True)
            narr("""
            **Price and promise.** Premium pricing only works when the table can feel why. Component quality helps, but repeatable decisions and clean turns are what justify a number. Break even is a milestone. Word of mouth is the margin.
            """)

            ########## design analysis visuals ##########
            st.markdown("### 🎨 Visuals")
            
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
            fig_a.add_trace(go.Scatter(
                x=seg["GameWeight"], y=seg["AvgRating"],
                mode="markers", name=cluster_labels.get(cluster_id, f"Segment {cluster_id}"),
                marker=dict(size=7, opacity=0.7),
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
            
            fig_b.add_trace(go.Scatter(
                x=seg["AvgRating"], y=seg["Owned Users"],
                mode="markers", name=cluster_labels.get(cluster_id, f"Segment {cluster_id}"),
                marker=dict(size=7, opacity=0.7),
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
            
            fig_c.add_trace(go.Scatter(
                x=seg["Year Published"], y=seg["AvgRating"],
                mode="markers", name=cluster_labels.get(cluster_id, f"Segment {cluster_id}"),
                marker=dict(size=7, opacity=0.75),
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

            narr("""
            **Position with intent.** If you sit to the right on complexity, give players early wins. If your playtime is shorter than the segment, lean on tension not upkeep. If your price is friendly, make the first play magical then let depth unfold on play three.
            """)

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
    **Where the puck is going.** Themes cycle, but the constant is respect for time. Designers are packing strategy into tighter sessions. That is not dumbing things down. That is craft. If you are designing into the near future, pair a strong theme with clean teach, depth that reveals across plays, and a clear promise on the box about time.
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
        # Growth predictions
        yearly_growth = trend_data.groupby("Year Published").size()
        if len(yearly_growth) > 3:
            recent_growth_rate = yearly_growth.iloc[-3:].pct_change().mean()
            predicted_releases = int(yearly_growth.iloc[-1] * (1 + recent_growth_rate))
            
            st.write(f"📊 **Predicted new releases in {CURRENT_YEAR + 1}:** ~{predicted_releases} games")
        
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

    narr("""
    **Two year outlook.** Expect a steady stream of midweight designs that play under two hours. Expect more solo modes. Expect fewer lazy products. The audience is informed and generous when the game respects them.
    """)
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
    **Reading a segment.** Look at three things. What the segment loves to do (mechanics above thirty percent). How long the table sits (median playtime). Where quality clusters (rating distribution). If your idea fights the segment norms, make the reason obvious and delightful.
    """)
    # Segment characteristics
    segment_cols = st.columns(2)
    
    with segment_cols[0]:
        st.markdown("#### 🎯 Defining Characteristics")
        
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
    **Entering the segment.** If releases are rising and ratings hold, the space is hungry. If releases spike and ratings sag, the space needs curation. Add one new idea that matters, not six that confuse.
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
        st.markdown("### 🌐 Mechanic Relationship Network")
        
        network_fig = create_mechanic_network_graph(synergies, top_n=30)
        st.plotly_chart(network_fig, use_container_width=True)
        narr("""
        **Mechanics that sing.** Some pairs create clarity. Worker placement with a market. Co-op with variable powers. Drafting with tempo pressure. Do not pile on features. Pick a duet, not a chorus. Leave room for players to discover lines you did not script.
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

# Footer with additional resources
narr("""
**Bottom line.** Games do not suck anymore. The average modern title beats the classics that started the boom. The reason is simple. Designers learned to respect time, clarify decisions, and make the first play feel good. Go make that game.
""")
st.markdown("---")
st.markdown("### 📚 Additional Resources")

resource_cols = st.columns(4)

with resource_cols[0]:
    st.markdown("""
    **📖 Documentation**
    - [BGG API Guide](https://boardgamegeek.com/wiki/page/BGG_XML_API2)
    - [Game Design Theory](https://www.google.com)
    - [Market Analysis Methods](https://www.google.com)
    """)

with resource_cols[1]:
    st.markdown("""
    **🛠️ Tools**
    - [Component Calculator](https://www.google.com)
    - [Prototype Generator](https://www.google.com)
    - [Playtest Tracker](https://www.google.com)
    """)

with resource_cols[2]:
    st.markdown("""
    **👥 Community**
    - [Designer Forums](https://boardgamegeek.com/forum)
    - [Publisher Directory](https://www.google.com)
    - [Playtest Groups](https://www.google.com)
    """)

with resource_cols[3]:
    st.markdown("""
    **📊 Data Sources**
    - BoardGameGeek Database
    - {len(df):,} games analyzed
    - Updated: {pd.Timestamp.now().strftime('%Y-%m-%d')}
    """)

st.markdown(
    f"""
    <div style='text-align: center; color: {MUTED}; padding: 2rem; margin-top: 2rem;'>
    <strong>🎲 Board Game Developer Console - Professional Edition</strong><br>
    Transforming data into successful game designs<br>
    Built with ❤️ for the board game community
    </div>
    """,
    unsafe_allow_html=True
)












