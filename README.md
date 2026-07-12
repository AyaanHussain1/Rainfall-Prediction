# Rainfall Prediction

A machine learning model that predicts whether rainfall will occur based on weather station readings — pressure, humidity, dew point, cloud cover, sunshine, wind direction, and wind speed.

## Overview

This project trains a Random Forest classifier, tuned via grid search, to predict same-day rainfall from standard meteorological measurements. It's a genuinely useful case study in a common real-world challenge: working with a small dataset (234 samples) and being transparent about the resulting overfitting risk, rather than only reporting the best-looking number.

## How It Works

1. **Data** — Used a weather dataset of 234 daily readings with 7 features (pressure, dew point, humidity, cloud cover, sunshine, wind direction, wind speed), evenly balanced between rainfall and no-rainfall days (117 each).
2. **Model Tuning** — Trained a **Random Forest Classifier**, tuned via `GridSearchCV` across 216 hyperparameter combinations (`n_estimators`, `max_depth`, `max_features`, `min_samples_split`, `min_samples_leaf`) with 5-fold cross-validation.
3. **Evaluation** — Evaluated on train, cross-validation, and a genuinely held-out test set:

   | Metric | Score |
   |---|---|
   | Train Accuracy | 93.0% |
   | Mean Cross-Validation Score (train) | 82.4% |
   | **Held-out Test Accuracy** | **68.1%** |

4. **Honest Finding** — The gap between training/cross-validation performance and the held-out test score is a clear sign of overfitting, driven primarily by the small dataset size (only 187 training samples) relative to the size of the hyperparameter search space. This is reported transparently rather than only highlighting the stronger-looking training numbers — a larger dataset would likely be needed to close this gap and get a model that's reliably production-ready.
5. **Predictive System** — Included a working example showing how to predict rainfall for a new set of weather readings.

## Tech Stack

- **Language:** Python
- **Data Handling:** Pandas, NumPy
- **Modeling:** Scikit-learn (Random Forest, GridSearchCV, cross-validation)
- **Visualization (EDA):** Matplotlib, Seaborn

## Project Structure

```
rainfall-prediction/
├── main.py           # Data loading, model tuning & evaluation
├── final_data.xls     # Weather dataset (234 daily readings)
└── requirements.txt    # Python dependencies
```

## Installation & Usage

```bash
# Clone the repository
git clone https://github.com/AyaanHussain1/rainfall-prediction.git
cd rainfall-prediction

# Install dependencies
pip install -r requirements.txt

# Run the training script
python main.py
```

## Results

| Metric | Score |
|---|---|
| Train Accuracy | 93.0% |
| Mean CV Score | 82.4% |
| Test Accuracy | 68.1% |

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| No Rainfall (0) | 0.72 | 0.57 | 0.63 |
| Rainfall (1) | 0.66 | 0.79 | 0.72 |

## License

This project is licensed under the [MIT License](LICENSE).

## Disclaimer

This project is built for educational and portfolio purposes. Given the small dataset size and resulting overfitting gap, it should not be used for real weather forecasting decisions.
