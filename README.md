# Crypto-Horizon
Welcome to **Crypto Horizon**, an Artificial Intelligence and Machine Learning predictive engine engineered to decode structural market liquidity  and forecast **Binance Coin (BNB/USD) price movements and directional velocity**, focusing on its **mature trend leading of Cryptocurrencies**.

This project moves past basic chart patterns to examine the deep structural correlations and liquidity linkages between **Binance Coin (BNB) and Bitcoin (BTC/USD)**. By analyzing cross-asset capital flows across a comprehensive decade-long dataset from kaggle, Crypto Horizon decodes the complex, shifting lead-lag dynamics that drive macro cryptocurrency cycles for passionate quantitative analysts and algorithmic traders.

## Project Directory Structure
| File Name | Description |
| :--- | :--- |
| `Crypto_Data_Since_2015.csv` | Historical multi-asset crypto index tracker |
| `Dependencies.py` | Environment settings, package alignments, seed locks |
| `AI_unsupervised.py` | Data cleansing, feature rolling windows, regime generation |
| `Correlation_engine.py` | Animated sweeping correlation matrix workspace |
| `Train_and_LSTM.py` | Chronological sequence construction, deep training matrix |
| `Model_evaluation.py` | Dashboard HUD server setup and metric telemetry |
| `README.md` | System documentation |

## Installation & Prerequisites
This system is tested on Python 3.10+ environments. Ensure you have the foundational scientific libraries and deep learning runtimes properly set up.
To initialize your development workspace, execute the following command:
`pip install numpy pandas scikit-learn tensorflow keras plotly dash seaborn matplotlib dash_chat openai`

## Code Module Walkthrough
**Data Ingestion & Unsupervised Clustering (AI_unsupervised.py)**
Loads raw time-series metrics into high-speed memory arrays and applies log-return transformation alongside structural rolling volatility filters. It isolates 3 separate hidden macro trends:
<img width="974" height="72" alt="image" src="https://github.com/user-attachments/assets/bf70006e-d6fe-4839-9c22-601ddbb849ae" />

## Time-series Windowing & Stacked LSTM (Train_and_LSTM.py)
MinMax dimensions normalisation maps price metrics cleanly to bound ranges before segmenting continuous historical streams into array matrices preserving target chronological structures. 
<img width="974" height="280" alt="image" src="https://github.com/user-attachments/assets/d2d0390f-64fb-4adb-8100-efd3fa953c70" />

## Production Deployment Telemetry (Model_evaluation.py)
Implements an analytical performance space using the Dash framework. Residual distributions can be modified live through interactive simulation modifiers to evaluate boundary limits and model performance:
**Execute directly from the terminal session to mount the live server app**
`Python Model_evaluation.py`

## Operational Telemetry Metrics
During active validation runtimes, your training pipelines output terminal statuses showing convergence behaviour:
--- LSTM Training Session ---
Train shapes - X: (1700, 10, 3), y: (1700,)
Test shapes  - X: (425, 10, 3), y: (425,)

--- Training TensorFlow/Keras LSTM Model Engine ---
Epoch 20/20 - loss: 0.000142 - val_loss: 0.000318
[SUCCESS] Fully Trained!
Final Model Performance -> Training Loss: 0.000142 | Validation Loss: 0.000318
