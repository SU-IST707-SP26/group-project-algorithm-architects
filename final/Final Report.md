# Tech Stock Return Prediction and Ranking System

## Team

Shivani Kankatala - ShivaniKankatala  
Poorvi Nidsoshi - poorvinidsoshi01  
Priyanka Pawar - ppawar03-byte  

---

## Introduction

Retail investors face a persistent problem: which stocks are worth buying this week, and which should be avoided? Professional tools are either inaccessible, expensive, or offer only qualitative "Buy/Hold/Sell" ratings rather than quantitative, ranked guidance. This project builds a machine-learning-based system that predicts weekly stock movement for a universe of stocks, ranks them from best to worst opportunity, and translates those predictions into concrete buy/avoid decisions with an estimated profit-and-loss outcome.

Our primary stakeholder is an **individual retail investor** managing a small portfolio, such as $10,000, who wants weekly, data-driven guidance. The stakeholder need is clear: the investor needs to know not only *which* stock is likely to outperform, but also *what to do about it* and *what the financial consequence of that action is likely to be*. Knowing that "NVDA is ranked #1" is not enough — the investor needs to know whether to buy it, avoid it, or treat it as neutral.

Our solution trains multiple model families — ARIMA, XGBoost, LSTM, and Elastic Net Logistic Regression — across 15 stocks spanning technology, energy, finance, healthcare, and consumer sectors. For each week, we rank all stocks by predicted movement strength, generate a buy/avoid signal for each stock, and simulate portfolio performance to assess whether the model's signals are financially useful. The full pipeline from raw data to ranked predictions to profit-and-loss simulation is implemented and evaluated on held-out test data from 2024–2026.

---

## Literature Review

### Prior Work and Stakeholder Context

Stock return prediction has a long history in both academic research and industry practice. Fischer & Krauss (2018) demonstrated that LSTM networks applied to the S&P 500 achieve roughly 55–60% directional accuracy — a meaningful edge over a 50/50 baseline. Sezer et al. (2020) reviewed deep learning approaches to financial time series and confirmed that technical indicators combined with sequence models generally outperform simpler baselines on short-horizon return prediction.

Professional tools used by institutional quants are proprietary and completely inaccessible to retail investors. Retail-facing tools like Robinhood show analyst consensus ratings ("Buy"/"Hold") but do not produce quantitative predictions or rankings, do not update weekly, and do not provide expected return estimates. A Schwab survey cited in our proposal found that 67% of active traders report difficulty identifying the best stock picks, while Vanguard research found that 72% struggle with when to adjust portfolio positions. These findings motivate a tool that outputs not only a ranking but also actionable signals.

### Why These Methods?

We chose complementary model families to capture different aspects of return predictability:

**ARIMA/ARIMAX** is the classical statistical baseline for time series. It models autocorrelation in returns and, with exogenous variables such as lagged features, can capture short-term momentum. Its weakness is the assumption of stationarity and linearity, which may miss complex interactions.

**XGBoost** is a gradient-boosted decision tree model that works well with structured tabular data. In the corrected version of this project, XGBoost was used as a multiclass classification model instead of a regression-only model. The model predicts whether each stock’s future 7-day movement belongs to one of three classes: Down, Neutral, or Up.

This correction was made because the earlier approach used predicted numeric returns and then converted them into positive or negative directions using a zero cutoff. That made the directional results unstable because very small returns close to zero were being treated as meaningful buy or avoid signals. The corrected XGBoost approach uses a threshold so that small movements near zero are classified as Neutral.

XGBoost was selected because it can capture nonlinear relationships between financial indicators such as lagged returns, moving averages, volatility, volume changes, RSI, and broader market index movement. Since stock behavior is rarely perfectly linear, a tree-based model is useful for identifying patterns across these engineered features.

**Elastic Net Logistic Regression** was used as a classification model to predict whether a stock’s 7-day future return would be positive or negative. Elastic Net combines L1 and L2 regularization, which helps reduce overfitting and handles correlated financial features better than standard logistic regression. This was useful because many technical indicators, such as moving averages, lagged returns, volatility, and volume-based features, are related to each other. Elastic Net also provides a simpler and more interpretable model compared with tree-based or deep learning models.

**LSTM** was included because stock data is sequential, and LSTM models are designed to learn patterns across time steps. LSTM can capture longer-term dependencies in time-series data that simpler models may miss.

**Random Forest** was included as another tree-based model for comparison. It is useful because it can handle nonlinear feature relationships and is generally more stable than a single decision tree.

### The Gap We Fill

No existing free tool ranks multiple stocks simultaneously by predicted return or movement strength, translates those rankings into buy/avoid decisions, and estimates the financial outcome of following those decisions. This project fills that gap.

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

**Coverage:** Approximately 2,500 daily observations per stock from 2015–2026, resampled to weekly frequency using Friday close for modeling

**Data quality:** Yahoo Finance is a licensed partner of NYSE/Nasdaq, data is adjusted for splits and dividends, and zero missing trading days were found in our download. The `yfinance` library is widely used in academic financial research, providing confidence in data integrity.

**Key observations from EDA:**

- Weekly returns for all stocks are approximately normally distributed with slight positive skew, centered near 0.
- Stocks are positively correlated, especially technology stocks, but magnitude differences across weeks are sufficient to distinguish ranking positions.
- NVDA shows the highest annualized volatility and the highest average weekly return over the study period.
- WMT and JNJ show the lowest volatility.
- Class balance shows that a trivial “always predict up” baseline can sometimes appear strong, which is why baseline comparison is important.

---

## Methods

### Preprocessing and Feature Engineering

All data was loaded from CSV files, date columns were converted to datetime format, and price/volume columns were cleaned of comma-formatting. Daily stock data was resampled to weekly frequency using Friday closing prices to reduce short-term noise.

The following feature families were constructed for each stock:

- **Lagged return features** to capture recent stock movement
- **Moving averages** to represent short-term and long-term price trends
- **Rolling volatility** to measure recent price risk and fluctuation
- **Volume-based features** to capture changes in trading activity
- **RSI and momentum indicators** to identify overbought/oversold behavior and recent trend direction
- **Market index features** from the S&P 500 and Nasdaq to include broader market movement

For **ARIMA/ARIMAX**, features were limited to lagged returns, moving averages, and volatility, which served as exogenous variables in SARIMAX.

For **XGBoost**, the engineered features were used as input variables to predict the 7-day future movement class for each stock. The corrected XGBoost model does not treat every small positive or negative return as a strong signal. Instead, the 7-day future return is converted into three classes using a threshold: Down, Neutral, and Up.

This allowed XGBoost to use technical indicators, lagged returns, moving averages, rolling volatility, RSI, volume-based features, and market index features to classify stock movement more realistically.

For **Elastic Net Logistic Regression**, similar engineered features were used, but the target was converted into a binary classification label. Since Elastic Net is a regularized linear model, using consistent and relevant features helped reduce overfitting and handle correlated indicators such as moving averages, lagged returns, volatility, and volume-based features.

---

### Target Variable

The main target variable was the **7-day future return** for each stock. This was calculated by comparing the current stock price with the stock price seven trading days ahead.

For **ARIMA/ARIMAX**, the target was treated as a regression problem because the model predicted a numeric future return.

For **XGBoost**, the corrected target was treated as a multiclass classification problem. The 7-day future return was first calculated and then converted into three classes:

- `0 = Down`: meaningful negative 7-day future return
- `1 = Neutral`: small movement close to zero
- `2 = Up`: meaningful positive 7-day future return

A threshold of approximately ±1% was used to define meaningful movement. This means that returns close to zero were not forced into either the Up or Down class. This correction made the XGBoost model more realistic because small market movements do not always represent useful trading signals.

For **Elastic Net Logistic Regression**, the 7-day future return was converted into a binary classification target:

- `1` = positive 7-day future return
- `0` = negative or zero 7-day future return

This allowed Elastic Net Logistic Regression to generate a direct buy/avoid signal for each stock.

---

### Train/Validation/Test Split

The data was split chronologically instead of randomly. This was important because stock prediction is time-based, and random splitting could allow future information to leak into the training process.

Older observations were used for training, while the most recent observations were reserved for validation and testing. This made the evaluation more realistic because the models were tested on future data that was not available during training.

The held-out test period focused on recent market data from 2024–2026, which helped evaluate whether the models could produce useful predictions in a realistic investment setting.

---

### Model Details

**ARIMA / ARIMAX:** ARIMA was used as a statistical time-series baseline. The single-stock ARIMA model was first applied to AAPL, and then ARIMAX was extended to all 15 stocks. The walk-forward ARIMAX approach used order (1,1,1). At each test step, the model was retrained on all available historical data and forecasted one step ahead. After each forecast, the actual value was added back into the history. This process was computationally expensive but realistic because it simulated what an investor would know at each point in time.

**XGBoost Multiclass Classification:** XGBoost was implemented as a multiclass classification model to predict the 7-day future movement class for all 15 stocks. The model used engineered features such as lagged returns, moving averages, rolling volatility, RSI, momentum indicators, volume-based features, calendar features, and market index features.

The previous version used `XGBRegressor` to predict numeric future returns and then converted those predictions into positive or negative directions. After feedback, this was corrected because the model output was being used like a classifier. The updated version uses `XGBClassifier`, which directly predicts one of three classes: Down, Neutral, or Up.

The Neutral class was important because it prevents very small returns near zero from being treated as strong buy or avoid signals. For example, a return close to 0% should not automatically be considered a meaningful upward or downward movement.

After predictions were generated, the model also produced class probabilities for each stock and date. These probabilities were used for ranking. Instead of ranking only by predicted return, the corrected ranking score was calculated as:

`Ranking Score = Probability of Up - Probability of Down`

For each date, all 15 stocks were ranked using this score. The stock with the highest score was selected as the strongest predicted opportunity for that period.

**Elastic Net Logistic Regression:** Elastic Net Logistic Regression was implemented as a classification model to predict whether each stock’s 7-day future return would be positive or negative. Unlike XGBoost, which predicts Down, Neutral, or Up, Elastic Net produced a positive or negative class prediction.

Elastic Net was selected because it combines L1 and L2 regularization. This helped reduce overfitting and made the model more stable when working with correlated financial indicators such as lagged returns, moving averages, volatility, RSI, momentum, and volume-based features. The model was evaluated using accuracy, precision, recall, F1 score, and the confusion matrix.

---

### Important Modeling Decisions and Issues Addressed

Several modeling decisions were made to keep the results reliable and useful.

**ARIMA / ARIMAX decisions:**

- ARIMA was used as a statistical baseline to compare against machine learning models.
- Walk-forward validation was used to avoid lookahead bias.
- ARIMAX included exogenous variables such as lagged returns, moving averages, and volatility.
- The model was retrained at each test step to simulate a realistic forecasting process.

**XGBoost decisions:**

- XGBoost was corrected from a regression-based approach to a multiclass classification approach.
- The previous version used `XGBRegressor`, but the corrected version uses `XGBClassifier`.
- The 7-day future return was converted into three classes: Down, Neutral, and Up.
- A threshold was used so that small returns close to zero were classified as Neutral instead of being forced into Up or Down.
- The same feature engineering process was applied across all 15 stocks so that stocks could be compared fairly.
- The model was evaluated using accuracy, balanced accuracy, macro precision, macro recall, macro F1 score, and a multiclass confusion matrix.
- Balanced accuracy and macro F1 score were included because normal accuracy can be misleading when one class dominates.
- Naive baselines were added, including an “always predict Up” baseline and a majority-class baseline.
- These baselines were important because if most stocks move upward in the dataset, a naive classifier can appear strong without learning useful patterns.
- XGBoost predictions were combined into one dataset so that each date could have a full stock ranking.
- The stock ranking was based on class probabilities using the score: `Probability of Up - Probability of Down`.
- The stock with the highest ranking score was selected as the top-ranked recommendation for portfolio simulation.

**Elastic Net Logistic Regression decisions:**

- Elastic Net was used as a classification model because it produced a direct positive/negative return signal.
- The 7-day future return was converted into a binary target.
- L1 and L2 regularization were used to reduce overfitting and handle correlated financial indicators.
- The model was evaluated using accuracy, precision, recall, F1 score, and the confusion matrix.
- Elastic Net was included because it is simpler and more interpretable than XGBoost, making it useful for explaining buy/avoid decisions.

**General decisions for all models:**

- Chronological splitting was used instead of random splitting to avoid lookahead bias.
- Models were trained on past data and tested on future observations.
- Evaluation included both prediction-error metrics and decision-based metrics.
- Model outputs were saved as CSV files so they could be reused for ranking, portfolio simulation, comparison, and final reporting.

---

### Supporting Files

| Notebook / File | Purpose |
|----------|---------|
| `arima_model-3.ipynb` | ARIMA/ARIMAX on AAPL; walk-forward, backtesting, trading signals |
| `Arima_For_15_Stocks.ipynb` | ARIMAX applied to all 15 stocks in a loop |
| `xgboost_15stocks_corrected_multiclass.ipynb` | Corrected XGBoost multiclass classification model for predicting Down, Neutral, and Up movement classes across 15 stocks |
| `xgboost_multiclass_predictions_15_stocks.csv` | Combined corrected XGBoost predictions with actual class, predicted class, and class probabilities |
| `xgboost_multiclass_metrics_15_stocks.csv` | Corrected XGBoost evaluation metrics including accuracy, balanced accuracy, macro precision, macro recall, and macro F1 score |
| `xgboost_multiclass_feature_importance_15_stocks.csv` | Feature importance output from the corrected XGBoost multiclass model |
| `xgboost_multiclass_top_ranked_portfolio.csv` | Portfolio/ranking output based on the corrected XGBoost probability-based ranking score |
| `elastic_net_logistic.ipynb` | Elastic Net Logistic Regression model for positive/negative return classification |

---

## Results

### Model Evaluation Framework

All models were evaluated on held-out test data from 2024–2026. Regression models such as ARIMA/ARIMAX were evaluated using MAE and RMSE on the return scale, along with directional interpretation where appropriate.

For the corrected XGBoost model, evaluation was performed as a multiclass classification problem. The model predicted whether each stock’s 7-day future movement was Down, Neutral, or Up. Because normal accuracy can be misleading when one class is more common, the XGBoost model was evaluated using accuracy, balanced accuracy, macro precision, macro recall, macro F1 score, and a multiclass confusion matrix.

Classification models such as Elastic Net Logistic Regression and LSTM were reported in classification terms with metrics such as accuracy, precision, recall, F1 score, and confusion matrices.

Baselines for comparison:

- **Random selection:** Expected to perform near chance level
- **Always Up baseline:** Always predicts the Up class
- **Majority class baseline:** Always predicts the most common class in the training or test data
- **Equal-weight buy-and-hold:** Invest equal amounts in all stocks each period
- **S&P 500 buy-and-hold:** Full-period compounded return

---

### ARIMA Results: Single-Stock AAPL and 15 Stocks

The walk-forward ARIMAX approach on AAPL produced:

- MAE and RMSE on test-set weekly returns
- Directional accuracy by comparing predicted versus actual return signs
- A backtesting simulation starting from $1,000 using BUY/SELL signals versus a buy-and-hold baseline

Directional accuracy for ARIMA is typically lower than XGBoost and LSTM because ARIMA assumes linearity. The walk-forward approach ensures no lookahead bias, but it is computationally expensive because the model is refit at each test step.

---

### XGBoost Results

XGBoost was evaluated as a multiclass classification model for predicting 7-day future movement across all 15 stocks. Instead of predicting only a numeric future return, the corrected model predicted whether each stock would move Down, remain Neutral, or move Up.

This correction was made after feedback on the original approach. The earlier version used `XGBRegressor` and then converted predicted returns into positive or negative directions using a zero cutoff. That approach could create misleading results because very small predicted returns close to zero were treated as meaningful signals. The corrected model uses a threshold-based class definition so that near-zero returns are classified as Neutral.

The XGBoost model was evaluated using accuracy, balanced accuracy, macro precision, macro recall, macro F1 score, and a multiclass confusion matrix. Balanced accuracy and macro F1 score were especially important because the dataset can be imbalanced. If one class appears more often than the others, normal accuracy alone may make the model look stronger than it actually is.

The model was also compared against naive baselines, including an “always predict Up” baseline and a majority-class baseline. This addressed the concern that if most stocks are positive during the test period, a simple model that always predicts Up could achieve high accuracy without learning useful financial patterns.

For stock ranking, XGBoost used predicted class probabilities. The corrected ranking score was calculated as:

`Ranking Score = Probability of Up - Probability of Down`

This score is more useful than a simple predicted return because it considers both upside potential and downside risk. For each date, all 15 stocks were ranked by this score, and the stock with the highest score was selected as the top-ranked recommendation for that period.

Overall, the corrected XGBoost model is more appropriate for the project goal because it directly predicts stock movement classes, avoids misleading near-zero signals, compares performance against naive baselines, and supports weekly ranking using probability-based scores.

---

### Elastic Net Logistic Regression Results

Elastic Net Logistic Regression was evaluated as a classification model. Instead of predicting the exact future return, it predicted whether the future 7-day return would be positive or negative.

The model was evaluated using accuracy, precision, recall, F1 score, and the confusion matrix. Accuracy measured the overall percentage of correct classifications. Precision showed how many predicted positive-return cases were actually positive. Recall showed how many actual positive-return cases the model successfully identified. F1 score provided a balance between precision and recall.

This model was useful because it gave a direct buy/avoid type signal. Compared with XGBoost, Elastic Net Logistic Regression was simpler and easier to interpret. However, because it is a linear model, it may not capture complex nonlinear market relationships as strongly as XGBoost.

---

### Summary Comparison

| Model | Stocks | Target | Primary Metric | Simulation |
|-------|--------|--------|----------------|------------|
| ARIMA / ARIMAX | 15 | Weekly return regression | MAE + directional accuracy | BUY/SELL backtesting |
| XGBoost | 15 | 7-day movement multiclass classification: Down, Neutral, Up | Accuracy, balanced accuracy, macro F1, confusion matrix | Weekly ranking using Probability of Up - Probability of Down |
| Elastic Net Logistic Regression | 15 | Positive vs. negative return classification | Accuracy, precision, recall, F1 score | BUY/AVOID signal evaluation |
| LSTM | 15 | Positive vs. negative return classification | Accuracy, precision, recall, F1 score | Classification-based signal evaluation |

---

## Discussion

### Were the Goals Achieved?

The project goal was achieved because the system moved beyond single-stock prediction and created a practical stock ranking and decision-support framework. The corrected XGBoost model helped classify each stock’s expected 7-day movement as Down, Neutral, or Up. This made the output more realistic because small returns close to zero were not treated as strong trading signals.

XGBoost also supported the ranking objective by using predicted class probabilities. For each week, stocks were ranked using the score `Probability of Up - Probability of Down`, which allowed the system to compare upside potential against downside risk. Elastic Net Logistic Regression supported the classification side by producing positive or negative return signals.

Together, these models helped answer the stakeholder’s main question: which stock should be considered for investment, which stocks should be avoided, and how confident the model is in that direction.

---

### Connection to Stakeholder Need

For a retail investor, the most useful output is not just a model score. The investor needs a simple and actionable result. The corrected XGBoost model supports this need by classifying stock movement into Down, Neutral, and Up classes, then ranking stocks using probability-based scores.

This makes the system practical because the final output can be interpreted as which stock is ranked highest, whether the model gives an Up, Neutral, or Down signal, and how the strategy performs compared with simple baselines.

---

## Limitations

**ARIMA scalability.** Walk-forward ARIMAX requires refitting at every test step, making it computationally prohibitive for large universes. The 15-stock ARIMA notebook is compute-intensive and was trained with limited hyperparameter tuning.

**XGBoost classification limitations.** The corrected XGBoost model is more appropriate than the earlier regression-to-direction approach, but it still has limitations. The threshold used to define Down, Neutral, and Up affects the class balance and the final evaluation results. If the threshold is too small, too many weak movements may be treated as signals. If the threshold is too large, too many observations may become Neutral. XGBoost can also capture nonlinear relationships, but it is less interpretable than simpler models. Feature importance helps identify influential variables, but it does not fully explain every individual prediction.

**Elastic Net linearity.** Elastic Net Logistic Regression is easier to interpret, but it assumes a mostly linear relationship between the input features and the probability of a positive return. Stock market behavior can be more complex than this.

**Market unpredictability.** Both models rely mainly on historical price, volume, and market-based features. Sudden news events, earnings results, macroeconomic changes, and unexpected market shocks are not fully captured in the dataset.

---

## Future Work

Future improvements can include stronger hyperparameter tuning for the corrected XGBoost classifier, testing different return thresholds for the Down, Neutral, and Up classes, additional feature selection for Elastic Net, and the use of more external signals such as earnings dates, analyst sentiment, macroeconomic indicators, and news sentiment. Another improvement would be to combine ARIMA, XGBoost, Elastic Net, and LSTM into an ensemble system so that each model contributes to the final stock ranking or signal.

The portfolio simulation can also be expanded by including transaction costs, risk-adjusted returns, stop-loss rules, and better position sizing instead of selecting only the top-ranked stock each week.

---

## References

Fischer, T., & Krauss, C. (2018). Deep learning with long short-term memory networks for financial market predictions. European Journal of Operational Research, 270(2), 654-669.

Sezer, O. B., Gudelek, M. U., & Ozbayoglu, A. M. (2020). Financial time series forecasting with deep learning. Applied Soft Computing, 90, 106181.

Yahoo Finance historical stock market data accessed using `yfinance`.

---

## Supporting Notebook Index

**Data:** All data is downloaded from Yahoo Finance using `yfinance`. A Google Drive link with cached CSVs is available at: *(add link here)*

---

## Presentation Explanation for XGBoost Correction

After the presentation feedback, the XGBoost section was corrected from a regression-based setup to a multiclass classification setup. In the earlier version, XGBoost predicted a numeric 7-day return, and then that value was converted into an up or down direction using zero as the cutoff. The issue with that approach is that a very small value close to zero could be treated as a real signal, even though it may not be meaningful in practice.

In the corrected version, XGBoost directly predicts three classes: Down, Neutral, and Up. A threshold is used so that small returns close to zero become Neutral. This avoids forcing every small movement into either an Up or Down prediction.

The corrected model also includes naive baselines, such as always predicting Up and always predicting the majority class. This is important because if the market is mostly going up during the test period, a naive classifier could get high accuracy without learning anything useful.

Because of this, the corrected evaluation uses balanced accuracy and macro F1 score, not only normal accuracy. For ranking, the model uses class probabilities and ranks stocks using:

`Ranking Score = Probability of Up - Probability of Down`

This makes the XGBoost result more realistic and better aligned with the project goal of ranking stocks based on meaningful expected movement.
