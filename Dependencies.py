# ========================================================================================
# CELL 1: PACKAGES AND ENVIRONMENT SETUP
# ========================================================================================
import os
import json
import plotly.io
import webbrowser
import numpy as np           # Used for handling arrays and mathematical arrays
import pandas as pd          # Used for loading and managing the dataset table
import seaborn as sns        # Used for drawing advanced statistical graphs and heatmaps
import tensorflow as tf      # Used for hardware acceleration and deep learning engine backend
import matplotlib.pyplot as plt    # Used for making the final prediction charts
import plotly.graph_objects as go  # Used for creating interactive and zoomable charts
import matplotlib.patches as mpatches # Used for drawing custom shapes and custom graph legends

from sklearn.preprocessing import MinMaxScaler  # Used to scale prices between 0 and 1
from sklearn.metrics import mean_squared_error, r2_score  # Used to measure prediction error
from sklearn.mixture import GaussianMixture     # Core AI clustering algorithm
from sklearn.tree import DecisionTreeClassifier # Used for rule-based classification models
from sklearn.cluster import KMeans    # Used for grouping data based on distance centers
from keras.models import Sequential   # Used to build the layer-by-layer model
from keras.layers import LSTM, Dense  # The deep learning layers we need

# Lock random numbers so you get the exact same results every time you run this cell
np.random.seed(42)
tf.random.set_seed(42)

print("Setup complete! & Ready to load your data.")
