Here’s a polished **README.md** you can use for your GitHub repository:

---

# Predicting Gold Prices in Chennai with AI (24K & 22K)

This project demonstrates how to **forecast Chennai gold prices (22K & 24K)** from 2012 to 2025 using state-of-the-art neural forecasting models: **TimeMixer, NHITS, and TSMixerx**. The workflow includes data preprocessing, feature engineering, model training, cross-validation, and evaluation with visualizations.

---

## **Dataset**

We use the **GoldRate – Chennai 2012–2025** dataset, containing:

* Daily gold rates for 22K and 24K gold
* Date information (day, month, year, day of the week)
* Engineered features: previous day rates (`prev_day_rates`) and 7-day moving averages (`MA_7`)

Example snippet:

| Date       | 22K Rate | 24K Rate | Month | Year | Day | DayOfWeek | Prev\_22K | Prev\_24K | MA\_7 |
| ---------- | -------- | -------- | ----- | ---- | --- | --------- | --------- | --------- | ----- |
| 2012-01-02 | 2560.0   | 2738.0   | 1     | 2012 | 2   | 0         | NaN       | NaN       | NaN   |
| 2012-01-03 | 2582.0   | 2762.0   | 1     | 2012 | 3   | 1         | 2560.0    | 2738.0    | NaN   |

---

## **Installation**

Install the latest version of **NeuralForecast** from GitHub:

```bash
!pip install git+https://github.com/Nixtla/neuralforecast.git
```

Other required packages:

```bash
pip install pandas matplotlib numpy
```

---

## **Data Preprocessing & Feature Engineering**

* Removed duplicates and handled missing values
* Engineered lag features (`prev_day_rates`) and rolling averages (`MA_7`)
* Extracted temporal features: day, month, year, day of the week
* Converted dataset to NeuralForecast format: `unique_id`, `ds`, `y`

---

## **Models**

We use three advanced neural forecasting models:

**TimeMixer** – Captures long-range dependencies by mixing past trends, acting like a “time-traveling detective.”

**NHITS** – Hierarchically organizes temporal information for multi-step forecasting, like stacking LEGO blocks.

**TSMixerx** – Transformer-inspired model detecting subtle sequential patterns, like spotting hidden melodies in historical trends.

---

## **Training & Cross-Validation**

* Forecast horizon: **120 days**
* Input size: **2000 historical points**
* Loss function: **MSE (Mean Squared Error)**
* Evaluation: **Rolling-window cross-validation** with 10 overlapping windows to simulate real-world forecasting and ensure robustness

---

## **Evaluation & Visualization**

We used the following metrics:

* **MAE (Mean Absolute Error)**
* **MSE (Mean Squared Error)**
* **SMAPE (Symmetric Mean Absolute Percentage Error)**

Example visualization of predictions:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10,8))
ax.plot(cv_preds_24["ds"], cv_preds_24["y"], label="Actual")
ax.plot(cv_preds_24["ds"], cv_preds_24["TimeMixer"], label="TimeMixer", ls="--")
ax.plot(cv_preds_24["ds"], cv_preds_24["NHITS"], label="NHITS", ls=":")
ax.plot(cv_preds_24["ds"], cv_preds_24["TSMixerx"], label="TSMixerx", ls="-.")
ax.set_xlabel("Date")
ax.set_ylabel("24K Gold Rate (INR)")
ax.legend()
fig.autofmt_xdate()
```

---

## **Results & Insights**

* **TimeMixer** and **NHITS** closely tracked actual gold prices
* **TSMixerx** captured long-term patterns but slightly underpredicted short-term spikes
* Ensemble predictions improved accuracy and robustness

---

## **Conclusion & Future Work**

* Neural forecasting models can reliably predict 22K and 24K gold prices in Chennai
* Future improvements:

  * Include macroeconomic indicators (USDINR, inflation, interest rates)
  * Hybrid classical + neural models for enhanced accuracy
  * Real-time updating for live forecasting
  * Explainable AI methods to interpret predictions

---

## **License**

This project is open-source and available under the MIT License.

---
