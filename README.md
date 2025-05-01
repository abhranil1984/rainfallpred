
## Overview

This project takes monthly meteorological data, creates rolling sequences, trains the attention network to learn spatio-temporal patterns, and then fits a Random Forest on the learned representations. It outputs forecasts, evaluation metrics, and visualizations (time series plots, attention heatmaps, feature importance charts).

---

## Features

- **Multi-Head Self-Attention** with positional encoding to capture temporal dependencies.
- **Enhanced Feature & Temporal Attention** modules to weigh input features and time steps dynamically.
- **Early Stopping & Learning Rate Scheduler** for robust attention network training.
- **Hybrid Pipeline**: Attention network for representation learning + Random Forest for final regression.
- **Comprehensive Visualizations**: Actual vs. predicted series, attention heatmaps, feature importances, error analysis.

---

## Getting Started

### Prerequisites

- Python 3.8+
- GPU with CUDA (optional but recommended for faster training)

Required Python packages (install via pip):

```bash
pip install pandas numpy scikit-learn matplotlib torch torchvision
```

### Installation

1. Clone this repository:
   ```bash
git clone https://github.com/abhranil1984/rainfallpred.git
cd rainfallpred
```

2. Install dependencies (see [Prerequisites](#prerequisites)).

---

## Usage

### Data Preparation

1. Place your input CSV named `DATA.csv` in the project root. It must have columns:
   - `PARAMETER`, `YEAR`, `REGION`, `JAN`–`DEC`, including precipitation in `PRECTOTCORR`.

2. The script will melt, pivot, encode, and scale features automatically.

### Training

Run the model script to train both the attention network and Random Forest:

```bash
python model.py
```

---

**Author:** Abhranil Sharma  
**Contact:** abhranil2001@gmail.com

