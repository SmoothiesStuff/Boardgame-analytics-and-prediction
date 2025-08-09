# streamlit_app.py (revised)
# Runs even on a fresh env by auto-installing missing packages.
# Keeps your existing functionality, plus a few hardening fixes.

import os
import sys
import subprocess

# -------------------------------
# Bootstrap: ensure required packages are installed
# -------------------------------
REQUIRED = [
    ("pandas", "pandas", None),
    ("numpy", "numpy", None),
    ("streamlit", "streamlit", None),
    ("plotly", "plotly", None),
    ("scikit-learn", "sklearn", None),
    ("joblib", "joblib", None),
]

# XGBoost is optional — we only attempt to import if any XGB models exist
OPTIONAL = [("xgboost", "xgboost", None)]


def _pip_install(package_spec: str):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_spec])
        return True
    except Exception:
        return False


def ensure_package(pkg_name: str, import_name: str, version: str | None):
    try:
        __import__(import_name)
        return True
    except Exception:
        spec = f"{pkg_name}=={version}" if version else pkg_name
        ok = _pip_install(spec)
        if ok:
            try:
                __import__(import_name)
                return True
            except Exception:
                return False
        return False


# Install required packages
_missing = []
for pkg_name, import_name, version in REQUIRED:
    if not ensure_package(pkg_name, import_name, version):
        _missing.append(pkg_name)

# Defer Streamlit import until after installation attempt
try:
    import numpy as np
    import pandas as pd
    import streamlit as st
    import plotly.express as px
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import pairwise_distances
    from joblib import load as joblib_load
except Exception as e:
    # If Streamlit isn't available we can't render UI, but try to print a helpful note
    print("Failed to import a required library after install attempts:", e)
    raise

# If anything failed to install, surface a warning in the UI but keep going if possible
if _missing:
    st.warning(
        "Some packages could not be auto-installed: " + ", ".join(sorted(set(_missing))) +
        " — please run: pip install " + " ".join(sorted(set(_missing)))
    )

# Conditionally load xgboost if any XGB models are present
HAS_XGB_MODELS = False

st.set_page_config(page_title="Board Game Developer Console", layout="wide")

# -------------------------------
# Config: columns and paths
# -------------------------------
EXCLUDE_FOR_CLUSTERING = [
    # post-release or target-ish
    "Owned Users", "LogOwned", "BayesAvgRating", "SalesPercentile",
    "Users Rated", "BGG Rank", "StdDev",
    # IDs and non-features
    "ID", "BGGId", "Name", "Description",
]

ALWAYS_NUMERIC_DEFAULT_ZERO = True

MODEL_PATHS = {
    "rating_rf": "models/rating_rf.joblib",
    "rating_xgb": "models/rating_xgb.joblib",
    "sales_rf": "models/sales_rf.joblib",
    "sales_xgb": "models/sales_xgb.joblib",
}

# Detect whether any XGB models exist; if so, ensure xgboost is available
for k in ("rating_xgb", "sales_xgb"):
    if os.path.exists(MODEL_PATHS.get(k, "")):
        HAS_XGB_MODELS = True
        break

if HAS_XGB_MODELS:
    ok = ensure_package("xgboost", "xgboost", None)
    if not ok:
        st.warning("XGBoost model files detected but 'xgboost' could not be installed. XGB predictions will be unavailable.")

# -------------------------------
# Helpers
# -------------------------------
@st.cache_data(show_spinner=False)
def load_df(uploaded_file):
    if uploaded_file is not None:
        try:
            return pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")
            st.stop()
    default_path = "cleaned_large_bgg_dataset.csv"
    if os.path.exists(default_path):
        try:
            return pd.read_csv(default_path)
        except Exception as e:
            st.error(f"Failed to read default CSV at {default_path}: {e}")
            st.stop()
    st.error("No dataset provided. Upload cleaned_large_bgg_dataset.csv or place it alongside this app.")
    st.stop()


def split_features(df: pd.DataFrame):
    # Cluster features = numeric columns excluding post-release/IDs
    X = df.drop(columns=[c for c in EXCLUDE_FOR_CLUSTERING if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=["number"])  # numeric only
    if ALWAYS_NUMERIC_DEFAULT_ZERO:
        X = X.fillna(0)
    # Keep non-feature columns for reporting
    keep_cols = [c for c in df.columns if c not in X.columns] + (["Name"] if "Name" in df.columns else [])
    keep_cols = [c for c in keep_cols if c in df.columns]
    meta = df[keep_cols].copy()
    return X, meta


@st.cache_resource(show_spinner=False)
def fit_clusterer(X: pd.DataFrame, k: int = 8, random_state: int = 42):
    if len(X) < k:
        st.warning("k is larger than the number of rows; reducing k.")
        k = max(2, int(max(2, len(X) // 2)))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Use robust default for broader sklearn compatibility
    try:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    except TypeError:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(X_scaled)
    return scaler, kmeans, pca, labels, coords


def year_percentile(series: pd.Series, value: float):
    # percent of values strictly below 'value'
    if series is None or len(series) == 0:
        return np.nan
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return np.nan
    return float((s < value).mean() * 100.0)


def build_input_vector(Xcols, profile_dict: dict):
    # Fill missing keys in profile with 0; keep only Xcols order
    filled = {c: 0 for c in Xcols}
    for k, v in profile_dict.items():
        if k in filled:
            filled[k] = v
    return pd.DataFrame([filled])[Xcols]


def nearest_neighbors_in_cluster(input_scaled, cluster_id, df_full, X, scaler, kmeans, topn=10):
    # filter same cluster
    X_scaled_full = scaler.transform(X)
    labels = kmeans.predict(X_scaled_full)
    mask = labels == cluster_id
    Xc = X_scaled_full[mask]
    dfc = df_full.loc[X.index[mask]].copy()

    dists = pairwise_distances(input_scaled, Xc)[0]
    dfc["__dist"] = dists
    return dfc.nsmallest(topn, "__dist")


@st.cache_resource(show_spinner=False)
def load_models(paths_dict: dict):
    models = {}
    for key, path in paths_dict.items():
        if os.path.exists(path):
            try:
                models[key] = joblib_load(path)
            except Exception as e:
                st.warning(f"Could not load {key} at {path}: {e}")
    return models


def predict_with_models(models: dict, X_input_df: pd.DataFrame):
    out = {}
    for key, model in models.items():
        try:
            out[key] = float(model.predict(X_input_df)[0])
        except Exception:
            out[key] = None
    return out


def rf_confidence_std(model, X_input_df: pd.DataFrame):
    # std of predictions across trees (proxy for uncertainty)
    try:
        ests = getattr(model, "estimators_", None)
        if ests is None:
            return None
        preds = np.column_stack([est.predict(X_input_df) for est in ests])
        return float(np.std(preds, axis=1)[0])
    except Exception:
        return None


# -------------------------------
# Sidebar: data + settings
# -------------------------------
st.sidebar.title("Data & Settings")
uploaded = st.sidebar.file_uploader("Upload cleaned_large_bgg_dataset.csv", type=["csv"])
df = load_df(uploaded)

st.sidebar.markdown("---")
k = st.sidebar.slider("K (clusters)", 2, 15, 8, step=1)
topn = st.sidebar.slider("Nearest neighbors to show", 3, 30, 10, step=1)
st.sidebar.markdown("---")
st.sidebar.caption("Optional: Place trained models in ./models or update paths in code.")

# -------------------------------
# Prepare data & cluster
# -------------------------------
X_all, meta = split_features(df)
scaler, kmeans, pca, labels, coords = fit_clusterer(X_all, k=k)

df_view = df.copy()
df_view["Cluster"] = labels
df_view["PCA1"] = coords[:, 0]
df_view["PCA2"] = coords[:, 1]

# -------------------------------
# Header
# -------------------------------
st.title("Board Game Developer Console")
st.write("Explore clusters, test a concept, find nearest neighbors, and (optionally) predict success.")

# -------------------------------
# CLUSTER MAP
# -------------------------------
st.subheader("📌 Cluster Map (PCA)")
color_by = st.selectbox("Color by:", ["Cluster", "Year Published"], index=0)
hover_cols = [c for c in ["Name", "Year Published", "BayesAvgRating", "Owned Users"] if c in df_view.columns]

fig = px.scatter(
    df_view,
    x="PCA1",
    y="PCA2",
    color=color_by,
    hover_data=hover_cols,
    title=f"PCA Projection (k={k})",
)
st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# CLUSTER SUMMARIES
# -------------------------------
st.subheader("📊 Cluster Summaries")
summary_cols = []
for c in ("BayesAvgRating", "Owned Users", "GameWeight", "Play Time"):
    if c in df_view.columns:
        summary_cols.append(c)

if summary_cols:
    cluster_summary = df_view.groupby("Cluster")[summary_cols].agg(["count", "mean", "median"]).round(2)
    st.dataframe(cluster_summary)
else:
    st.info("No summary columns available (e.g., BayesAvgRating / Owned Users).")

# -------------------------------
# INPUT FORM: New Game Concept
# -------------------------------
st.subheader("🧪 Test a New Game Concept")

with st.form("concept_form"):
    # Core known fields
    c1, c2, c3, c4 = st.columns(4)
    year_published = c1.number_input("Year Published", value=2023, step=1)
    min_players = c2.number_input("Min Players", value=2, step=1)
    max_players = c3.number_input("Max Players", value=4, step=1)
    play_time = c4.number_input("Play Time (min)", value=60, step=5)

    c5, c6, c7, c8 = st.columns(4)
    min_age = c5.number_input("Min Age", value=10, step=1)
    weight = c6.number_input("Complexity (GameWeight)", value=2.5, step=0.1, format="%.1f")
    kickstarted = c7.selectbox("Kickstarted", ["No", "Yes"], index=0)
    best_players = c8.number_input("BestPlayers (if known, else 0)", value=0, step=1)

    # Mechanic and Theme toggles based on columns present
    mech_cols = sorted([c for c in X_all.columns if c.startswith("Mechanic_")])
    theme_like_cols = [
        c for c in X_all.columns
        if c.startswith("Theme_") or c in {"Fantasy", "Adventure", "Horror", "Science Fiction", "Space Exploration", "Animals", "Sports"}
    ]
    theme_cols = sorted(theme_like_cols)

    st.markdown("**Select mechanics and themes:**")
    selected_mechs = st.multiselect("Mechanics", mech_cols[:100], default=[])
    selected_themes = st.multiselect("Themes", theme_cols[:100], default=[])

    # Advanced numeric fields (optional)
    st.markdown("**Optional numeric fields:** (leave blank to default to 0)")
    adv_cols = [
        "ComAgeRec", "LanguageEase", "NumWant", "NumWish",
        "MfgPlaytime", "ComMinPlaytime", "ComMaxPlaytime", "MfgAgeRec",
        "NumAlternates", "NumExpansions", "NumImplementations",
        "IsReimplementation",
    ]
    adv_vals = {}
    a1, a2, a3, a4 = st.columns(4)
    for i, col in enumerate(adv_cols):
        container = [a1, a2, a3, a4][i % 4]
        adv_vals[col] = container.number_input(col, value=0, step=1)

    submitted = st.form_submit_button("Analyze Concept")

if submitted:
    # Build profile dict
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

    # turn on mechanics/themes
    for m in selected_mechs:
        profile[m] = 1
    for t in selected_themes:
        profile[t] = 1

    # Align to training feature set
    Xcols = list(X_all.columns)
    x_input = build_input_vector(Xcols, profile)

    # Scale and cluster assign
    x_scaled = scaler.transform(x_input)
    assigned_cluster = int(getattr(kmeans, "predict")(x_scaled)[0])

    st.success(f"Assigned to Cluster **{assigned_cluster}**")

    # Nearest neighbors from same cluster
    neighbors = nearest_neighbors_in_cluster(
        x_scaled, assigned_cluster, df_view, X_all, scaler, kmeans, topn=topn
    )

    # Build year-relative stats for neighbors
    rows = []
    has_year = "Year Published" in df_view.columns
    for _, r in neighbors.iterrows():
        year = int(r.get("Year Published", 0))
        same_year = df_view[df_view["Year Published"] == year] if has_year else pd.DataFrame()

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
    st.subheader("🔎 Nearest Neighbors (same cluster)")
    st.dataframe(nn_df, use_container_width=True)

    # Download neighbors
    st.download_button(
        "Download neighbors as CSV",
        data=nn_df.to_csv(index=False).encode("utf-8"),
        file_name="nearest_neighbors.csv",
        mime="text/csv",
    )

    # Optional: predictions (if models exist)
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

        # RF confidence proxy if RF models are available
        conf_cols = st.columns(2)
        rf_rating = models.get("rating_rf")
        rf_sales = models.get("sales_rf")
        if rf_rating is not None:
            r_std = rf_confidence_std(rf_rating, x_input)
            conf_cols[0].caption(
                f"RF rating σ (lower≈more confident): {r_std:.3f}" if r_std is not None else "RF rating σ: n/a"
            )
        if rf_sales is not None:
            s_std = rf_confidence_std(rf_sales, x_input)
            conf_cols[1].caption(
                f"RF sales σ (lower≈more confident): {s_std:.3f}" if s_std is not None else "RF sales σ: n/a"
            )

# -------------------------------
# BONUS: Cluster Explorer
# -------------------------------
st.markdown("---")
st.subheader("🔭 Cluster Explorer")
cluster_pick = st.selectbox("Choose a cluster to inspect", sorted(df_view["Cluster"].unique()))
dfc = df_view[df_view["Cluster"] == cluster_pick].copy()

left, right = st.columns([2, 1])
with left:
    fig2 = px.scatter(
        dfc,
        x="PCA1",
        y="PCA2",
        hover_data=hover_cols,
        title=f"Cluster {cluster_pick} games (PCA)",
    )
    st.plotly_chart(fig2, use_container_width=True)
with right:
    cols_to_show = [c for c in ["Name", "Year Published", "BayesAvgRating", "Owned Users", "GameWeight", "Play Time"] if c in dfc.columns]
    st.dataframe(dfc[cols_to_show].head(50), use_container_width=True)

# -------------------------------
# FOOTER
# -------------------------------
st.caption("Tip: Adjust k on the sidebar. Upload your most recent cleaned dataset for fresh clusters.")
