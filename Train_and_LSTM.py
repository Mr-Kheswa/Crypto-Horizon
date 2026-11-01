# =========================================================================================
# CELL 4 & 5: DEEP LEARNING SESSION (TRAIN-TEST SPLIT & LSTM NETWORK MODELLING)
# =========================================================================================
from Dependencies import *      # Import specific functions required from Dependencies
from AI_unsupervised import *   # Contains clean_df with 'Market_Regime_Cluster'

print("--- LSTM Training Session ---")

# Feature Engineering & Selection
# We use the asset prices alongside the unsupervised AI market states to guide the LSTM network
features_list = ["BNB", "BTC", "Market_Regime_Cluster"]
dataset = clean_df[features_list].values

# Data Normalization
# Scale all features between 0 and 1 to prevent gradient explosion and optimize LSTM training
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Time Series Windowing Structure (Lookback Window Formulation)
def create_sequences(data, lookback=10):
    X, y = [], []
    for i in range(len(data) - lookback):
        # Extract previous window step records across all features
        X.append(data[i : i + lookback, :])
        # Target variable: Next day's BNB Price (Index 0 in our feature matrix)
        y.append(data[i + lookback, 0])
    return np.array(X), np.array(y)

lookback_window = 10
X, y = create_sequences(scaled_data, lookback=lookback_window)

# Sequential Train-Test Split (Chronological Order Preservation)
# 80% Training for model stabilization, 20% Testing for validation
split_index = int(len(X) * 0.80)

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

print(f"Train shapes - X: {X_train.shape}, y: {y_train.shape}")
print(f"Test shapes  - X: {X_test.shape}, y: {y_test.shape}")

# Deep Learning LSTM Architecture Design
# Built using the TensorFlow/Keras framework to capture long-term sequential dependencies
model = Sequential()

# Layer 1: First LSTM layer with 50 units, returning sequences for stacked architecture
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))

# Layer 2: Second LSTM layer with 50 units, processing sequential state contexts
model.add(LSTM(units=50, return_sequences=False))

# Layer 3: Dense output layer mapping hidden dimensions back to a single continuous price point
model.add(Dense(units=1))

# Compile the neural engine utilizing Adam optimizer and Mean Squared Error loss tracking
model.compile(optimizer="adam", loss="mean_squared_error")

print("\n--- Training TensorFlow/Keras LSTM Model Engine ---")
# Train the model with early stopping or standard epochs while keeping batch dimensions uniform
history = model.fit(
    X_train, 
    y_train, 
    epochs=20, 
    batch_size=32, 
    validation_data=(X_test, y_test),
    verbose=1
)

# Model Evaluation & Performance Analytics
train_loss = history.history['loss'][-1]
val_loss = history.history['val_loss'][-1]
print(f"\n[SUCCESS] Fully Trained!")
print(f"Final Model Performance -> Training Loss: {train_loss:.6f} | Validation Loss: {val_loss:.6f}")