Tech Stock Return Prediction and Ranking System

Team

Shivani Kankatala - ShivaniKankatala 
Poorvi Nidsoshi - poorvinidsoshi01 
Priyanka Pawar - ppawar03-byte 

---

## Introduction

Retail investors face a persistent problem: which stocks are worth buying this week, and which should be avoided? Professional tools are either inaccessible, expensive, or offer only qualitative "Buy/Hold/Sell" ratings rather than quantitative, ranked guidance. This project builds a machine-learning-based system that predicts weekly returns for a universe of stocks, ranks them from best to worst opportunity, and translates those predictions into concrete buy/avoid decisions with an estimated profit-and-loss outcome.

Our primary stakeholder is an **individual retail investor** managing a small portfolio (e.g., $10,000) who wants weekly, data-driven guidance. The stakeholder need is clear: the investor needs to know not only *which* stock is likely to outperform, but also *what to do about it* and *what the financial consequence of that action is likely to be*. Knowing that "NVDA is ranked #1" is not enough — the investor needs to know whether to buy it, and what gain or loss they might expect if they follow that signal.

Our solution trains multiple model families — ARIMA, XGBoost, LSTM, and Elastic Net Logistic Regression — across 15 stocks spanning technology, energy, finance, healthcare, and consumer sectors. For each week, we rank all stocks by predicted 7-day return, generate a buy/avoid signal for each stock, and simulate portfolio performance to assess whether the model's signals are financially useful. The full pipeline from raw data to ranked predictions to profit-and-loss simulation is implemented and evaluated on held-out test data from 2024–2026.

---

## Literature Review

### Prior Work and Stakeholder Context

Stock return prediction has a long history in both academic research and industry practice. Fischer & Krauss (2018) demonstrated that LSTM networks applied to the S&P 500 achieve roughly 55–60% directional accuracy — a meaningful edge over a 50/50 baseline. Sezer et al. (2020) reviewed deep learning approaches to financial time series and confirmed that technical indicators combined with sequence models generally outperform simpler baselines on short-horizon return prediction.

Professional tools used by institutional quants are proprietary and completely inaccessible to retail investors. Retail-facing tools like Robinhood show analyst consensus ratings ("Buy"/"Hold") but do not produce quantitative predictions or rankings, do not update weekly, and do not provide expected return estimates. A Schwab survey cited in our proposal found that 67% of active traders report difficulty identifying the best stock picks, while Vanguard research found that 72% struggle with when to adjust portfolio positions. These findings motivate a tool that outputs not only a ranking but also actionable signals.

### Why These Methods?

We chose three complementary model families to capture different aspects of return predictability:

**ARIMA/ARIMAX** is the classical statistical baseline for time series. It models autocorrelation in returns and, with exogenous variables (lagged features), can capture short-term momentum. Its weakness is the assumption of stationarity and linearity, which may miss complex interactions.
**Gradient**
**XGBoost** is a gradient-boosted decision tree model that works well with structured tabular data. It was used in this project as a regression model to predict the 7-day future return for 15 stocks across technology, finance, healthcare, energy, and consumer/industrial sectors. XGBoost was selected because it can capture nonlinear relationships between financial indicators such as lagged returns, moving averages, volatility, volume changes, RSI, and broader market index movement. This makes it useful for stock prediction, where the relationship between past behavior and future return is rarely perfectly linear.
**Elastic Net Logistic Regression** was used as a classification model to predict whether a stock’s 7-day future return would be positive or negative. Elastic Net combines L1 and L2 regularization, which helps reduce overfitting and handles correlated financial features better than standard logistic regression. This was useful because many technical indicators, such as moving averages, lagged returns, volatility, and volume-based features, are related to each other. Elastic Net also provides a simpler and more interpretable model compared with tree-based or deep learning models.
**LSTM** 
**Random Forest**

### The Gap We Fill

No existing free tool ranks multiple stocks simultaneously by predicted return, translates those rankings into buy/avoid decisions, and estimates the financial outcome of following those decisions. This project fills that gap.

---

## Data and Methods

### Data

**Source:** Yahoo Finance via `yfinance`

**Universe:** 15 stocks spanning five sectors:
- *Technology:* AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA
- *Finance:* JPM, GS
- *Healthcare:* JNJ, PFE
- *Energy:* XOM, CVX
- *Consumer/Industrial:* WMT, BA

**Market Indices:** S&P 500 (^GSPC), Nasdaq (^IXIC) — used as exogenous features for market context

**Coverage:** ~2,500 daily observations per stock (2015–2026), resampled to weekly (Friday close) for modeling

**Data quality:** Yahoo Finance is a licensed partner of NYSE/Nasdaq, data is adjusted for splits and dividends, and zero missing trading days were found in our download. The `yfinance` library is widely used in academic financial research, providing confidence in data integrity.

**Key observations from EDA:**
- Weekly returns for all stocks are approximately normally distributed with slight positive skew, centered near 0
- Stocks are positively correlated (tech stocks especially), but magnitude differences across weeks are sufficient to distinguish ranking positions
- NVDA shows the highest annualized volatility (~45%) and the highest average weekly return over the study period; WMT and JNJ show the lowest volatility
- Class balance (% of weeks with positive return) ranges from ~51% (BA) to ~58% (NVDA), confirming that a trivial "always predict up" baseline would score near 50–55% directional accuracy

### Methods

#### Preprocessing and Feature Engineering

All data was loaded from CSV files, date columns converted to datetime, and price/volume columns cleaned of comma-formatting. Daily data was resampled to weekly frequency using Friday closing prices to reduce noise.

The following feature families were constructed for each stock:



For ARIMA, features were limited to lagged returns, moving averages, and volatility, which serve as exogenous variables in SARIMAX.

#### Target Variable


#### Train/Validation/Test Split



#### Model Details

**ARIMA (single-stock and 15-stock):** Walk-forward ARIMAX with order (1,1,1). At each test step, the model is retrained on all available history and forecasts one step ahead, then the actual value is added to history. This is computationally expensive but realistic — it simulates what an investor would have access to each week. Exogenous variables are lagged return features.


#### Important Modeling Decisions and Issues Addressed


#### Supporting Files

| Notebook | Purpose |
|----------|---------|
| `arima_model-3.ipynb` | ARIMA/ARIMAX on AAPL; walk-forward, backtesting, trading signals |
| `Arima_For_15_Stocks.ipynb` | ARIMAX applied to all 15 stocks in a loop |

---

## Results

### Model Evaluation Framework

All models were evaluated on held-out test data (2024–2026). Regression models (ARIMA, XGBoost) are reported in terms of **MAE** and **RMSE** on the return scale (i.e., percentage points), as well as **directional accuracy**, **precision**, **recall**, and **F1 score** derived by thresholding predictions at zero. Classification models (LSTM, Elastic Net) are reported directly in classification terms with confusion matrices.

Baselines for comparison:
- **Random selection:** Expected ~50% directional accuracy
- **Equal-weight buy-and-hold:** Invest equal amounts in all stocks each period
- **S&P 500 buy-and-hold:** Full-period compounded return

### ARIMA Results (Single-Stock AAPL and 15 Stocks)

The walk-forward ARIMAX approach on AAPL produced:
- MAE and RMSE on test-set weekly returns (2025–2026)
- Directional accuracy: predictions were evaluated by comparing predicted vs. actual return signs
- A **backtesting simulation** starting from $1,000 using BUY/SELL signals (buy if predicted return > 0, otherwise sell/short) versus a buy-and-hold baseline

Directional accuracy for ARIMA is typically lower than XGBoost and LSTM because ARIMA assumes linearity. The walk-forward approach ensures no lookahead bias, at the cost of significant computation (re-fitting at each test step).

### XGBoost Results

XGBoost was evaluated as a regression model for predicting 7-day future returns across all 15 stocks. The model produced numeric predicted returns for each stock and each test date.

The main regression metrics used were MAE and RMSE. MAE measured the average absolute prediction error, while RMSE gave more weight to larger prediction errors. In addition to regression metrics, the predicted returns were converted into positive or negative return directions. This allowed the model to be evaluated using directional accuracy, precision, recall, and F1 score.

The XGBoost predictions were also used for stock ranking. For each date, all 15 stocks were ranked from highest predicted return to lowest predicted return. The top-ranked stock was selected as the model’s recommendation for that period. A portfolio simulation was then performed using the top-ranked selections to evaluate how the strategy performed over time.

Overall, XGBoost was useful because it supported both prediction and ranking. It did not only predict whether a stock may go up or down, but also helped compare all stocks in the same week and identify the strongest predicted opportunity.

### Elastic Net Logistic Regression Results

Elastic Net Logistic Regression was evaluated as a classification model. Instead of predicting the exact future return, it predicted whether the future 7-day return would be positive or negative.

The model was evaluated using accuracy, precision, recall, F1 score, and the confusion matrix. Accuracy measured the overall percentage of correct classifications. Precision showed how many predicted positive-return cases were actually positive. Recall showed how many actual positive-return cases the model successfully identified. F1 score provided a balance between precision and recall.

This model was useful because it gave a direct buy/avoid type signal. Compared with XGBoost, Elastic Net Logistic Regression was simpler and easier to interpret. However, because it is a linear model, it may not capture complex nonlinear market relationships as strongly as XGBoost.

### Summary Comparison

| Model | Stocks | Target | Primary Metric | Simulation |
|-------|--------|--------|----------------|------------|
| ARIMA / ARIMAX | 15 | Weekly return regression | MAE + directional accuracy | BUY/SELL backtesting |
| XGBoost | 15 | 7-day future return regression | MAE, RMSE, directional accuracy | Weekly ranking + top-stock portfolio |
| Elastic Net Logistic Regression | 15 | Positive vs. negative return classification | Accuracy, precision, recall, F1 score | BUY/AVOID signal evaluation |
| LSTM | 15 | Positive vs. negative return classification | Accuracy, precision, recall, F1 score | Classification-based signal evaluation |

---

## Discussion

### Were the Goals Achieved?
The project goal was achieved because the system moved beyond single-stock prediction and created a practical stock ranking and decision-support framework. XGBoost helped generate predicted returns for all 15 stocks, which made it possible to rank stocks from strongest to weakest opportunity each week. Elastic Net Logistic Regression supported the classification side by producing positive or negative return signals.

Together, these models helped answer the stakeholder’s main question: which stock should be considered for investment, and which stocks should be avoided?

### Connection to Stakeholder Need
For a retail investor, the most useful output is not just a model score. The investor needs a simple and actionable result. XGBoost supports this need by ranking stocks based on predicted return, while Elastic Net Logistic Regression supports it by giving a positive or negative return signal.

This makes the system practical because the final output can be interpreted as which stock is ranked highest, whether the model gives a buy or avoid signal, what return the model expects, and how the strategy performs compared with simple baselines.

---

## Limitations


**ARIMA scalability.** Walk-forward ARIMAX requires refitting at every test step, making it computationally prohibitive for large universes. The 15-stock ARIMA notebook is compute-intensive and was trained with limited hyperparameter tuning.

**XGBoost interpretability.** XGBoost can capture nonlinear relationships, but it is less interpretable than simpler models. Feature importance helps identify which variables influenced the model, but it does not fully explain every individual prediction.

**Elastic Net linearity.** Elastic Net Logistic Regression is easier to interpret, but it assumes a mostly linear relationship between the input features and the probability of a positive return. Stock market behavior can be more complex than this.

**Market unpredictability.** Both models rely mainly on historical price, volume, and market-based features. Sudden news events, earnings results, macroeconomic changes, and unexpected market shocks are not fully captured in the dataset.
---

## Future Work
Future improvements can include stronger hyperparameter tuning for XGBoost, additional feature selection for Elastic Net, and the use of more external signals such as earnings dates, analyst sentiment, macroeconomic indicators, and news sentiment. Another improvement would be to combine ARIMA, XGBoost, Elastic Net, and LSTM into an ensemble system so that each model contributes to the final stock ranking or signal.

The portfolio simulation can also be expanded by including transaction costs, risk-adjusted returns, stop-loss rules, and better position sizing instead of selecting only the top-ranked stock each week.

---

## References


---

## Supporting Notebook Index

**Data:** All data is downloaded from Yahoo Finance using `yfinance`. A Google Drive link with cached CSVs is available at: *(add link here)*
