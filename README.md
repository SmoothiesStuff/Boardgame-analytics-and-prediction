# Boardgame-analytics-and-prediction
data processing, ML, analytics, and a Streamlit app on Boardgame data from BGG

Access Dashboard at: 
https://boardgame-analytics-and-prediction-enmcp3dmzdqkqtebyztjoe.streamlit.app/

Github repo:
https://github.com/SmoothiesStuff/Boardgame-analytics-and-prediction

Text (208)608-3279 if the dashboard crashes so I can reboot it. 

# Board Game Developer Console

########## What this is ###########
A Streamlit app that turns the BGG 2023 dataset into practical design and market intelligence. It helps you spot promising niches, understand mechanic synergies, and estimate success.

########## Core features ###########
* Opportunity Finder with a weighted Opportunity Score
* Interactive clustering of game segments with PCA view
* Market overview charts for rating, complexity, playtime, age, ownership
* Segment Deep Dive tables with top games and mechanics
* Model Lab for fast success estimation from feature combos
* One click CSV exports of your filtered data and segment lists

########## Data you need ###########
* A cleaned BGG dataset CSV. Example columns:
  * Name, Year Published, Min Players, Max Players, Play Time, Min Age
  * Users Rated, Rating Average, Complexity Average, Owned Users
  * Mechanics and Categories as one hot or multi value columns
* File name suggestion: cleaned_large_bgg_dataset.csv placed in the project root or data/

########## Install ###########
```bash
python -m venv .venv
source .venv/bin/activate         # on Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```

Minimal requirements.txt
```
streamlit
pandas
numpy
scikit-learn
scipy
plotly
streamlit-plotly-events
joblib
```

Optional
```
python-dotenv
```

########## Run ###########
```bash
export STREAMLIT_SERVER_FILE_WATCHER_TYPE=poll
export WATCHDOG_FORCE_POLLING=true
streamlit run streamlit_app.py     # or: streamlit run app.py
```

########## Quick start ###########
1) Launch the app  
2) Use the sidebar to pick dataset file and default filters  
3) Click Apply  
4) Explore tabs in order: Overview, Clusters, Deep Dive, Opportunity, Model Lab, Export

########## UI map ###########
* Sidebar
  * Dataset selector and basic filters
  * Global toggles for mechanics, categories, time window
  * Reset filters
* Overview tab
  * Ratings vs Complexity scatter with filter highlighting
  * Playtime and Age bands with quality overlays
  * Ownership distributions as market sanity check
* Clusters tab
  * KMeans on engineered features
  * PCA 2D map with click to inspect cluster contents
* Deep Dive tab
  * Per segment tables: size, avg rating, ownership, mechanics, exemplar games
* Opportunity tab
  * Ranked list of segments with Opportunity Score and a short rationale
* Model Lab tab
  * Simple predictors for success likelihood from chosen features
* Export tab
  * Download current filters, top segments, and selected tables as CSV

########## Opportunity Score ###########
Purpose
* Find where healthy demand meets limited recent supply

Inputs per segment or category
* Growth3y — releases count growth over last 3 years
* AvgRating — normalized to 10 point scale
* MedianOwned — proxy for audience size
* Saturation — recent releases density in this segment
* SuccessRate — share of games with rating at least 7

Suggested normalization
* Min–max to 0..1 for each metric within candidate segments
* For Saturation use inverse saturation (invSat = 1 − norm(Saturation))

Weights
* Growth3y — 0.30
* AvgRating — 0.25
* MedianOwned — 0.20
* invSat — 0.15
* SuccessRate — 0.10

Equation
```
OppScore_0_1 = 0.30*Growth3y' + 0.25*AvgRating' + 0.20*MedianOwned' + 0.15*invSat' + 0.10*SuccessRate'
OppScore_pct = round(100 * OppScore_0_1)
```

Interpretation
* High score means strong quality signals, real audience, low recent crowding, and upward release trend

########## How to use the analytics ###########
* Find a lane
  * Sort Opportunity tab by score, open top segments, read mechanics bundle and exemplar games
* Pressure test a concept
  * In Model Lab select mechanics, playtime, complexity, age. Check predicted success and similar clusters
* Sanity check market fit
  * In Overview confirm your target sits in quality-dense regions of complexity and playtime
* Build a shortlist
  * Export the Deep Dive table for the best segments and study the common patterns

########## Configuration knobs ###########
* Clustering
  * KMeans k — default 6 to 10
  * Features include rating, complexity, playtime bins, age, ownership, mechanics vectors
* Filters
  * Year window, min ratings count, min ownership, target mechanics and categories
* Scoring
  * Adjust weights in config to match your studio goals

########## Reproducibility ###########
* Random seeds are fixed in clustering and PCA
* All exports include filter and parameter metadata in the header row

########## Extend the project ###########
* Swap in a newer BGG dump and rerun cleaning
* Add new engineered features — time since release, velocity, mechanic diversity
* Plug in alternative models in Model Lab — logistic regression, gradient boosting

########## Troubleshooting ###########
* Plotly not installed — add plotly to requirements.txt and reinstall
* Widgets crashing on defaults — ensure default selections exist in current dataset
* Long app reloads on file change — keep the polling env vars set as shown above

########## Repo layout ###########
```
project-root/
  streamlit_app.py
  requirements.txt
  data/
    cleaned_large_bgg_dataset.csv
  models/
    vectorizers.joblib
  exports/
    ...generated csv files...
  config/
    app.yaml             # optional weights, thresholds
```

########## License and citation ###########
* Academic course project for data visualization and analytics  
* Cite BGG as the original data source when publishing findings

