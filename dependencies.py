"""
Centralized third-party imports and shared configuration for the Crypto-Horizon
project. Every module below imports from here instead of repeating boilerplate
imports — this replaces the original notebook's `from Dependencies import *`
wildcard pattern with an explicit, well-defined module.
"""
# Configure TensorFlow before importing it to suppress routine startup messages.
import os

# Keep TensorFlow's startup log spam down; only errors are shown.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Shared data-science, plotting, clustering, scaling, and metric libraries used by pipeline modules.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# Shared Keras model, layer, and callback classes used to define/train the LSTM.
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Apply one consistent plotting style to figures created across the project.
sns.set_theme(style="whitegrid")
