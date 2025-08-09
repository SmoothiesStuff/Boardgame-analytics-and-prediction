# streamlit_app.py (updated to use local Parquet by default)

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from joblib import load as joblib_load

st.set_page_config(page_title="Board Game Developer Console", layout="wide")

DEFAULT_DATA_PATH = "cleaned_large_bgg_dataset.parquet"

EXCLUDE_FOR_CLUSTERING = [
    'Owned Users', 'LogOwned', 'BayesAvgRating', 'SalesPercentile',
    'Users Rated', 'BGG Rank', 'StdDev',
    'ID', 'BGGId', 'Name', 'Description'
]

ALWAYS_NUMERIC_DEFAULT_ZERO = True

MODEL_PATHS = {
    "rating_rf": "models/rating_rf.joblib",
    "rating_xgb": "models/rating_xgb.joblib",
    "sales_rf": "models/sales_rf.joblib",
    "sales_xgb": "models/sales_xgb.joblib",
}

@st.cache_data(show_spinner=True)
def load_df(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    if os.path.exists(DEFAULT_DATA_PATH):
        return pd.read_parquet(DEFAULT_DATA_PATH)
    st.error("No dataset provided. Upload a CSV or include cleaned_large_bgg_dataset.parquet.")
    st.stop()

# The rest of your original code follows, unchanged...
# (Keep all the clustering, nearest neighbor, and prediction logic exactly as before)
