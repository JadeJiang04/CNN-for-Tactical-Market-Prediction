# CNN for Tactical Market Prediction

Hull Tactical Market Prediction Competition Solution

## Overview

This project uses 1D Convolutional Neural Networks (1D-CNN) to predict daily excess returns of the S&P 500 index for the Hull Tactical Market Prediction competition. The goal is to maximize the volatility-adjusted Sharpe ratio while controlling volatility, validating machine learning approaches in financial time series prediction.

## Key Features

- **1D-CNN**: Captures local patterns and short-term dependencies in financial time series
- **Multi-dimensional Features**: Covers market, economic, interest rate, valuation, volatility, and sentiment indicators
- **Robust Regularization**: Handles high-noise financial data and prevents overfitting
- **Trading Signals**: Generates portfolio weights in the 0-2 range as required by the competition

## Dataset

- **Training Data**: ~9,000 trading days with 100+ original features
- **Feature Categories**:
  - M*: Market indicators
  - E*: Economic indicators
  - I*: Interest rates
  - P*: Price/valuation metrics
  - V*: Volatility measures
  - S*: Sentiment indicators
  - MOM*: Momentum factors
  - D*: Dummy variables

### Selected Features (24)

```
M1, M10, M11, M12, M13,      # Market
E1, E10, E11,               # Economic
I1, I2, I3, I4,             # Interest rates
P1, P10, P11, P12, P13,     # Valuation
V1, V10, V11,               # Volatility
S1, S10, S11,               # Sentiment
term_spread                 # I2 - I1
```

## Model Architecture

| Layer                 | Specification                               |
| --------------------- | ------------------------------------------- |
| Input                 | (30, 24) - 30-day lookback × 24 features    |
| Conv1D + BN + Dropout | 32 filters, kernel=3, ReLU, 20% dropout     |
| Conv1D + BN + GAP     | 64 filters, kernel=3, ReLU, Global Avg Pool |
| Dropout               | 30%                                         |
| Dense + Dropout       | 32 neurons, ReLU, 20% dropout               |
| Output                | 1 neuron, linear activation                 |

## Training

| Parameter     | Value                          |
| ------------- | ------------------------------ |
| Batch Size    | 32                             |
| Epochs        | 30 (early stopping patience=5) |
| Learning Rate | 0.001                          |
| Optimizer     | Adam                           |
| Loss          | MSE                            |
| Validation    | 10% time-series split          |

## Signal Generation

Convert model predictions to trading signals in [0, 2] range:

```python
signal = clip(prediction × 400 + 1, 0.0, 2.0)
```

## Results

### Training Performance

- Best Validation MSE: 0.00011313
- Reduction: >99.2% from initial value
- No significant overfitting observed

### Competition Scores

| Model                | Kaggle Public Score |
| -------------------- | ------------------- |
| Full-feature CNN     | 0.724               |
| Simplified CNN       | 1.038               |
| Competition Baseline | 0.466               |

## Project Structure

```
.
├── TACTICAL MARKET PREDICTION.ipynb    # Main code
├── train.csv                             # Training data
├── test.csv                              # Test data
├── kaggle_score.png                      # Score screenshot
├── report.pdf                            # Project report
└── README.md                             # This file
```

## Requirements

```
python>=3.11
polars
numpy
pandas
tensorflow>=2.0
scikit-learn
kaggle
```
