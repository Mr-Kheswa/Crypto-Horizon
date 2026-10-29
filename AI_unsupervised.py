# =========================================================================================
# CELL 2: UNSUPERVISED AI ANALYSIS (DATA INGESTION, BINARY LABELLING & TREE CLASSIFIER)
# =========================================================================================
from Dependencies import * # Import specific functions required from Dependencies

csv_filename = "Crypto_Data_Since_2015.csv"
print("--- Loading Specific Columns ---")

# Load and standardize headers to lowercase right away
df_raw = pd.read_csv(csv_filename)
df_raw.columns = df_raw.columns.str.strip().str.lower()

# Extract features directly without complex background processing loops
clean_df = pd.DataFrame()
clean_df["Date"] = pd.to_datetime(df_raw["date"])
clean_df["BTC"] = pd.to_numeric(df_raw["bitcoin (usd)"], errors="coerce")
clean_df["BNB"] = pd.to_numeric(df_raw["binance coin (bnb)"], errors="coerce")

# Instantly drop the huge raw data frame from your computer's RAM to protect the kernel
del df_raw

# Sort sequentially and remove missing data entries
clean_df = clean_df.sort_values("Date").set_index("Date").dropna()

# -----------------------------------------------------------------------------------------
# UNSUPERVISED AI DATA ANALYSIS — BINANCE COIN REGIME CLUSTERING
# -----------------------------------------------------------------------------------------
print("--- Extracting Features & Training Unsupervised AI Classifier ---")

# Compute historical daily price changes (Log Returns)
clean_df["BTC_Returns"] = np.log(clean_df["BTC"] / clean_df["BTC"].shift(1))
clean_df["BNB_Returns"] = np.log(clean_df["BNB"] / clean_df["BNB"].shift(1))

# FOCUS ACCENT: Shift structural rolling window tracker to monitor BNB Volatility instead of BTC
clean_df["BNB_Volatility"] = clean_df["BNB_Returns"].rolling(window=10).std()

# Clear mathematical lag row offsets
clean_df = clean_df.dropna()

# Select features for unsupervised learning focused on BNB behavior alongside BTC context
unsupervised_features = clean_df[["BNB_Returns", "BTC_Returns", "BNB_Volatility"]].values

# Train a highly efficient Unsupervised K-Means model to find 3 distinct BNB market regimes
ai_clusterer = KMeans(n_clusters=3, random_state=42, n_init="auto")
clean_df["Market_Regime_Cluster"] = ai_clusterer.fit_predict(unsupervised_features)

# Extract distance to cluster centers to serve as a BNB AI Anomaly/Signal Score
cluster_distances = ai_clusterer.transform(unsupervised_features)
clean_df["AI_Signal_Score"] = np.min(cluster_distances, axis=1)

print(f"Unsupervised AI Preprocessing Complete! Kernel completely safe.")
print(f"Ready for analysis arrays: {len(clean_df)} continuous records.")
clean_df[["BNB", "BTC", "BNB_Volatility", "Market_Regime_Cluster", "AI_Signal_Score"]].head()