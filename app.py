# streamlit_app.py — Parquet‑first, pretty earth‑tone visuals, interactive UX
# Designed for Streamlit Community Cloud (public link ready)
# --------------------------------------------------------
# Highlights
# - Loads local Parquet by default (CSV/Parquet upload supported)
# - Clean clustering workflow with PCA projection
# - Interactive filters (year, weight, play time) that update plots live
# - Polished earth‑tone color palette + subtle styling
# - Rich charts: scatter + density contours, cluster summaries, neighbor table
# - Optional model predictions (if joblib files exist)

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
# Page config & light styling
# ---------------------------------
st.set_page_config(
    page_title="Board Game Developer Console",
    page_icon="🎲",
    layout="wide",
)

# Earth‑tone palette (colorblind friendly leaning)
PALETTE = [
    "#5A6A62",  # sage
    "#A67C52",  # ochre/bronze
    "#7C8C4A",  # olive
    "#C4A484",  # sand
    "#4F4538",  # espresso
    "#B56576",  # clay rose
    "#6B705C",  # moss
    "#D4A373",  # wheat
    "#8D6B94",  # muted plum
    "#708090",  # slate
]

ACCENT = "#A67C52"  # headings
MUTED  = "#5A6A62"
BG_SOFT = "#FAF7F2"

# Inject a little CSS to soften the look
st.markdown(
    f"""
    <style>
      .stApp {{ background-color: {BG_SOFT}; }}
      section[data-testid="stSidebar"] {{ background-color: #F3EFE7; }}
      h1, h2, h3, h4 {{ color: {ACCENT}; }}
      .metric-small span {{ color: {MUTED}; font-size: 0.9rem; }}
      .css-ffhzg2 p, .stMarkdown p {{ color: #3a3a3a; }}
      .earthcard {{ border: 1px solid #e5dcc9; border-radius: 12px; padding: 0.75rem 1rem; background: #FFFCF7; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------
# Config: paths & columns
# ---------------------------------
DEFAULT_PARQUET_PATH = "cleaned_large_bgg_dataset.parquet"
DEFAULT_CSV_PATH     = "cleaned_large_bgg_dataset.csv"

# Exclude columns that are post‑release or IDs for clustering
EXCLUDE_FOR_CLUSTERING = [
    "Owned Users", "LogOwned", "BayesAvgRating", "SalesPercentile",
    "Users Rated", "BGG Rank", "StdDev",
    "ID", "BGGId", "Name", "Description",
]

ALWAYS_NUMERIC_DEFAULT_ZERO = True

MODEL_PATHS = {
    "rating_rf": "models/rating_rf.joblib",
    "rating_xgb": "models/rating_xgb.joblib",
    "sales_rf":  "models/sales_rf.joblib",
    "sales_xgb": "models/sales_xgb.joblib",
}

# ---------------------------------
# Data loading helpers
# ---------------------------------
@st.cache_data(show_spinner=True)
def load_df(uploaded_file) -> pd.DataFrame:
    """Load data from upload (CSV/Parquet) or fall back to repo files."""
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
    """Return (X numeric feature matrix, meta columns for reporting)."""
    X = df.drop(columns=[c for c in EXCLUDE_FOR_CLUSTERING if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=["number"])  # numeric only
    if ALWAYS_NUMERIC_DEFAULT_ZERO:
        X = X.fillna(0)
    keep_cols = [c for c in df.columns if c not in X.columns]
    if "Name" in df.columns:  # ensure Name present if exists
        keep_cols = list(dict.fromkeys(keep_cols + ["Name"]))
    meta = df[keep_cols].copy()
    return X, meta


@st.cache_resource(show_spinner=False)
def fit_clusterer(X: pd.DataFrame, k: int = 8, random_state: int = 42):
    if len(X) < k:
        st.warning("k is larger than the number of rows; reducing k.")
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


def rf_confidence_std(model, X_input_df: pd.DataFrame):
    try:
        ests = getattr(model, "estimators_", None)
        if ests is None:
            return None
        preds = np.column_stack([est.predict(X_input_df) for est in ests])
        return float(np.std(preds, axis=1)[0])
    except Exception:
        return None

# ---------------------------------
# Sidebar — data & global settings
# ---------------------------------
st.sidebar.title("Data & Settings")
uploaded = st.sidebar.file_uploader("Upload dataset (CSV or Parquet)", type=["csv", "parquet"])
df = load_df(uploaded)

st.sidebar.markdown("---")
# Provide sensible defaults if columns are missing
year_col = "Year Published" if "Year Published" in df.columns else None
min_year, max_year = (int(df[year_col].min()), int(df[year_col].max())) if year_col else (2010, 2024)

weight_col = "GameWeight" if "GameWeight" in df.columns else None
min_w, max_w = (float(df[weight_col].min()), float(df[weight_col].max())) if weight_col else (1.0, 5.0)

ptime_col = "Play Time" if "Play Time" in df.columns else None
min_t, max_t = (int(df[ptime_col].min()), int(df[ptime_col].max())) if ptime_col else (10, 240)

k = st.sidebar.slider("K (clusters)", 2, 15, 8, step=1)
topn = st.sidebar.slider("Nearest neighbors to show", 3, 30, 10, step=1)

# Filters
st.sidebar.subheader("Filters")
yr_rng = st.sidebar.slider("Year range", min_year, max_year, (min(max(min_year, 2010), max_year), max_year))
wt_rng = st.sidebar.slider("Complexity (weight)", float(math.floor(min_w)), float(math.ceil(max_w)), (max(1.0, min_w), min(5.0, max_w)))
pt_rng = st.sidebar.slider("Play time (min)", min_t, max_t, (min_t, max_t))

st.sidebar.caption("Optional: Place trained models in ./models (joblib). Parquet is preferred for speed.")

# ---------------------------------
# Prepare data & cluster
# ---------------------------------
X_all, meta = split_features(df)
scaler, kmeans, pca, labels, coords = fit_clusterer(X_all, k=k)

# Build a working view with projections
view = df.copy()
view["Cluster"] = labels
view["PCA1"] = coords[:, 0]
view["PCA2"] = coords[:, 1]

# Apply filters (if columns exist)
mask = pd.Series(True, index=view.index)
if year_col:
    mask &= view[year_col].between(yr_rng[0], yr_rng[1])
if weight_col:
    mask &= view[weight_col].between(wt_rng[0], wt_rng[1])
if ptime_col:
    mask &= view[ptime_col].between(pt_rng[0], pt_rng[1])

view_f = view[mask].copy()

# ---------------------------------
# Header
# ---------------------------------
st.title("Board Game Developer Console")
st.write("Explore clusters, test a concept, find nearest neighbors, and (optionally) predict success.")

# Small KPIs
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Games in view", f"{len(view_f):,}")
with c2:
    st.metric("Clusters", f"{len(np.unique(labels))}")
with c3:
    if "BayesAvgRating" in view_f.columns:
        st.metric("Median Bayes rating", f"{float(view_f['BayesAvgRating'].median()):.2f}")
    else:
        st.metric("Median Bayes rating", "n/a")
with c4:
    if "Owned Users" in view_f.columns:
        st.metric("Median owners", f"{int(view_f['Owned Users'].median()):,}")
    else:
        st.metric("Median owners", "n/a")

# ---------------------------------
# Tabs for main workflows
# ---------------------------------
_tab1, _tab2, _tab3 = st.tabs(["🗺️ Cluster Map", "🧪 Concept Tester", "🔭 Cluster Explorer"])  # map, concept, explorer

# -----------------
# Tab 1: Cluster Map
# -----------------
with _tab1:
    st.subheader("📌 PCA Projection")

    color_by_opts = ["Cluster"]
    if year_col: color_by_opts.append(year_col)
    color_by = st.selectbox("Color by", color_by_opts, index=0, key="colorby_map")

    hover_cols = [c for c in ["Name", year_col, "BayesAvgRating", "Owned Users", weight_col, ptime_col] if c and c in view_f.columns]

    fig = px.scatter(
        view_f,
        x="PCA1", y="PCA2",
        color=color_by,
        color_discrete_sequence=PALETTE,
        hover_data=hover_cols,
        height=580,
        title=f"PCA Projection (k={k})",
    )
    fig.update_traces(marker=dict(size=8, opacity=0.8, line=dict(width=0)))
    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))

    # Add density contours for context (based on filtered view)
    if len(view_f) > 50:
        fig2 = px.density_contour(view_f, x="PCA1", y="PCA2", color_discrete_sequence=["#8c7a64"])  # soft brown
        for tr in fig2.data:
            tr.update(opacity=0.25)
            fig.add_trace(tr)

    st.plotly_chart(fig, use_container_width=True)

    # Quick per‑cluster summary bar (if owners/rating available)
    st.markdown("<div class='earthcard'>", unsafe_allow_html=True)
    cols = st.columns(2)
    if "Owned Users" in view_f.columns:
        owners = view_f.groupby("Cluster")["Owned Users"].median().reset_index()
        owners.columns = ["Cluster", "Median Owners"]
        bar1 = px.bar(owners, x="Cluster", y="Median Owners", color="Cluster",
                      color_discrete_sequence=PALETTE, height=320,
                      title="Cluster median owners")
        bar1.update_layout(showlegend=False, margin=dict(l=10, r=10, t=60, b=10))
        cols[0].plotly_chart(bar1, use_container_width=True)
    if "BayesAvgRating" in view_f.columns:
        rating = view_f.groupby("Cluster")["BayesAvgRating"].median().reset_index()
        rating.columns = ["Cluster", "Median Rating"]
        bar2 = px.bar(rating, x="Cluster", y="Median Rating", color="Cluster",
                      color_discrete_sequence=PALETTE, height=320,
                      title="Cluster median rating")
        bar2.update_layout(showlegend=False, margin=dict(l=10, r=10, t=60, b=10))
        cols[1].plotly_chart(bar2, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------
# Tab 2: Concept Tester
# -----------------
with _tab2:
    st.subheader("🧪 Test a New Game Concept")

    # Prepare feature lists
    X_cols = list(X_all.columns)
    mech_cols = sorted([c for c in X_cols if c.startswith("Mechanic_")])
    theme_cols = sorted([
        c for c in X_cols
        if c.startswith("Theme_") or c in {"Fantasy", "Adventure", "Horror", "Science Fiction", "Space Exploration", "Animals", "Sports"}
    ])

    with st.form("concept_form"):
        c1, c2, c3, c4 = st.columns(4)
        year_published = c1.number_input("Year Published", value=min(2023, max_year), step=1)
        min_players = c2.number_input("Min Players", value=2, step=1)
        max_players = c3.number_input("Max Players", value=4, step=1)
        play_time    = c4.number_input("Play Time (min)", value=60, step=5)

        c5, c6, c7, c8 = st.columns(4)
        min_age    = c5.number_input("Min Age", value=10, step=1)
        weight     = c6.number_input("Complexity (GameWeight)", value=2.5, step=0.1, format="%.1f")
        kickstarted = c7.selectbox("Kickstarted", ["No", "Yes"], index=0)
        best_players = c8.number_input("BestPlayers (if known, else 0)", value=0, step=1)

        st.markdown("**Select mechanics and themes:**")
        m1, m2 = st.columns(2)
        selected_mechs  = m1.multiselect("Mechanics", mech_cols[:100], default=[])
        selected_themes = m2.multiselect("Themes", theme_cols[:100], default=[])

        st.markdown("**Optional numeric fields:** (leave blank for 0)")
        adv_cols = [
            "ComAgeRec","LanguageEase","NumWant","NumWish",
            "MfgPlaytime","ComMinPlaytime","ComMaxPlaytime","MfgAgeRec",
            "NumAlternates","NumExpansions","NumImplementations",
            "IsReimplementation",
        ]
        adv_vals = {}
        a1, a2, a3, a4 = st.columns(4)
        for i, col in enumerate(adv_cols):
            container = [a1, a2, a3, a4][i % 4]
            adv_vals[col] = container.number_input(col, value=0, step=1)

        submitted = st.form_submit_button("Analyze Concept")

    if submitted:
        profile = {
            "Year Published": year_published,
            "Min Players": min_players,
            "Max Players": max_players,
            "Play Time": play_time,
            "Min Age": min_age,
            "GameWeight": weight,
            "Kickstarted": 1 if kickstarted == "Yes" else 0,
            "BestPlayers": best_players,
        }
        profile.update(adv_vals)
        for m in selected_mechs: profile[m] = 1
        for t in selected_themes: profile[t] = 1

        x_input = build_input_vector(X_cols, profile)
        x_scaled = scaler.transform(x_input)
        assigned_cluster = int(kmeans.predict(x_scaled)[0])
        st.success(f"Assigned to Cluster **{assigned_cluster}**")

        # Nearest neighbors in same cluster
        neighbors = nearest_neighbors_in_cluster(
            x_scaled, assigned_cluster, view, X_all, scaler, kmeans, topn=topn
        )

        # Compose neighbor summary table
        rows = []
        has_year = year_col is not None
        for _, r in neighbors.iterrows():
            year = int(r.get(year_col, 0)) if has_year else 0
            same_year = view[view[year_col] == year] if has_year else pd.DataFrame()
            bayes = r.get("BayesAvgRating", np.nan)
            owned = r.get("Owned Users", np.nan)
            bayes_pct = year_percentile(same_year.get("BayesAvgRating", pd.Series(dtype=float)), bayes) if has_year else np.nan
            owned_pct = year_percentile(same_year.get("Owned Users", pd.Series(dtype=float)), owned) if has_year else np.nan
            rows.append({
                "Name": r.get("Name", "Unknown"),
                "Year": year,
                "Cluster": int(r.get("Cluster", -1)),
                "Dist": round(float(r["__dist"]), 4),
                "Bayes Rating": None if pd.isna(bayes) else round(float(bayes), 2),
                "Better Than X% (Rating, same year)": None if pd.isna(bayes_pct) else f"{bayes_pct:.0f}%",
                "Owned Users": None if pd.isna(owned) else int(owned),
                "Better Than X% (Sales, same year)": None if pd.isna(owned_pct) else f"{owned_pct:.0f}%",
            })
        nn_df = pd.DataFrame(rows)

        # Layout: neighbors + mini visual
        left, right = st.columns([2, 1])
        with left:
            st.subheader("🔎 Nearest Neighbors (same cluster)")
            st.dataframe(nn_df, use_container_width=True)
            st.download_button(
                "Download neighbors as CSV",
                data=nn_df.to_csv(index=False).encode("utf-8"),
                file_name="nearest_neighbors.csv",
                mime="text/csv",
            )
        with right:
            # A tiny concept vs cluster radar on core knobs (if present)
            rad_feats = [f for f in [weight_col, ptime_col] if f]
            if len(rad_feats) >= 1:
                cluster_slice = view[view["Cluster"] == assigned_cluster]
                stats = {}
                for f in rad_feats:
                    stats[f] = {
                        "concept": float(x_input.iloc[0][f]) if f in x_input.columns else np.nan,
                        "cluster_med": float(pd.to_numeric(cluster_slice[f], errors="coerce").median()),
                    }
                categories = list(stats.keys())
                concept_vals = [stats[c]["concept"] for c in categories]
                cluster_vals = [stats[c]["cluster_med"] for c in categories]
                # Normalize ranges for nicer polar view
                def norm(vals):
                    v = pd.Series(vals).astype(float)
                    lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
                    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
                        return [1 for _ in v]
                    return list((v - lo) / (hi - lo) * 100)
                fig_r = go.Figure()
                fig_r.add_trace(go.Scatterpolar(r=norm(concept_vals), theta=categories, fill='toself', name='Concept', line=dict(color=ACCENT)))
                fig_r.add_trace(go.Scatterpolar(r=norm(cluster_vals), theta=categories, fill='toself', name='Cluster median', line=dict(color=MUTED)))
                fig_r.update_layout(height=360, margin=dict(l=10,r=10,t=40,b=10), showlegend=True)
                st.plotly_chart(fig_r, use_container_width=True)

        # Optional model predictions
        st.subheader("📈 Predicted Success (if models available)")
        models = load_models(MODEL_PATHS)
        if not models:
            st.info("No models found. Place trained models in ./models or adjust MODEL_PATHS in code.")
        else:
            preds = predict_with_models(models, x_input)
            cols = st.columns(2)
            with cols[0]:
                rrf = preds.get("rating_rf")
                rxg = preds.get("rating_xgb")
                st.markdown("**Rating Models**")
                st.write(f"RandomForest: {rrf:.2f}" if rrf is not None else "RandomForest: n/a")
                st.write(f"XGBoost: {rxg:.2f}" if rxg is not None else "XGBoost: n/a")
            with cols[1]:
                srf = preds.get("sales_rf")
                sxg = preds.get("sales_xgb")
                st.markdown("**Sales Models**")
                st.write(f"RandomForest: {int(round(srf)):,}" if srf is not None else "RandomForest: n/a")
                st.write(f"XGBoost: {int(round(sxg)):,}" if sxg is not None else "XGBoost: n/a")
            conf_cols = st.columns(2)
            rf_rating = models.get("rating_rf")
            rf_sales  = models.get("sales_rf")
            if rf_rating is not None:
                r_std = rf_confidence_std(rf_rating, x_input)
                conf_cols[0].caption(f"RF rating σ (lower≈more confident): {r_std:.3f}" if r_std is not None else "RF rating σ: n/a")
            if rf_sales is not None:
                s_std = rf_confidence_std(rf_sales, x_input)
                conf_cols[1].caption(f"RF sales σ (lower≈more confident): {s_std:.3f}" if s_std is not None else "RF sales σ: n/a")

# -----------------
# Tab 3: Cluster Explorer
# -----------------
with _tab3:
    st.subheader("🔭 Explore a Cluster")
    cluster_pick = st.selectbox("Choose a cluster to inspect", sorted(view["Cluster"].unique()))
    dfc = view[view["Cluster"] == cluster_pick].copy()

    left, right = st.columns([2, 1])
    with left:
        fig2 = px.scatter(
            dfc, x="PCA1", y="PCA2",
            color="Cluster",
            color_discrete_sequence=PALETTE,
            hover_data=[c for c in ["Name", year_col, "BayesAvgRating", "Owned Users", weight_col, ptime_col] if c and c in dfc.columns],
            title=f"Cluster {cluster_pick} games (PCA)", height=560,
        )
        fig2.update_traces(marker=dict(size=9, opacity=0.85))
        fig2.update_layout(showlegend=False, margin=dict(l=10,r=10,t=60,b=10))
        st.plotly_chart(fig2, use_container_width=True)

    with right:
        cols_to_show = [c for c in ["Name", year_col, "BayesAvgRating", "Owned Users", "GameWeight", "Play Time"] if c and c in dfc.columns]
        st.dataframe(dfc[cols_to_show].head(80), use_container_width=True)

# ---------------------------------
# Footer
# ---------------------------------
st.caption("Tip: Upload a dataset or include cleaned_large_bgg_dataset.parquet in the repo for instant use.")
