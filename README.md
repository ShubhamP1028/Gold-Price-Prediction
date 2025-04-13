
# Gold Price Prediction using Linear Regression

This repository contains a project for predicting gold prices using a Linear Regression model. The project involves data cleaning, feature engineering, feature scaling, and model training using historical gold price data.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![Linear-Regression](https://img.shields.io/badge/Linear--Regression-Model-red?logo=Linear-regression)
  

- `linear-regression`
- `machine-learning`
- `regression-model`
- `gold-price-prediction`
- `data-science`
- `scikit-learn`


## 📁 Dataset
The dataset contains historical gold stock data with the following columns:
- `Date`
- `Open`
- `High`
- `Low`
- `Close` (Target Variable)
- `Volume`

## ⚙️ Project Structure
```
.
├── goldstock.csv                  # Raw dataset

├── gold_data_cleaning.py         # Script for data cleaning and feature engineering

├── goldstock_cleaned.csv         # Cleaned and processed dataset

├── linear_regression_model.ipynb # Jupyter notebook for training and evaluation (optional)

├── README.md                     # Project documentation
```

## 🔍 Features Engineered
- `price_change = Close - Open`
- `high_low_spread = High - Low`
- `day_of_week` (0 = Monday, ..., 6 = Sunday)
- `month`

All features (excluding the target) are scaled using StandardScaler for optimal model performance.

## 🚀 How to Run
1. Clone the repository
```bash
git clone https://github.com/yourusername/gold-price-prediction.git
cd gold-price-prediction
```
2. Install dependencies (if any)
```bash
pip install pandas scikit-learn
```
3. Run data cleaning script
```bash
python gold_data_cleaning.py
```
4. Train the model using your own notebook or script.

## 📈 Model
We use **Linear Regression** to predict the `Close` price based on engineered features.

---

## 📈 Model Performance

| Metric     | Score      |
|------------|------------|
| Accuracy   | 0.8979%    |
| Precision  | 0.8987%    |
| Recall     | 0.8979%    |
| R^2        | 1.6        |
| MAE        | 9.48       |           
| MSE        | 186.65     |

> Distribution of residuals `/visuals`

![image](https://github.com/user-attachments/assets/14a4b7b4-a368-4c22-81d5-66fa520f247f)

> Actual vs Prediction `/visuals`

![Screenshot 2025-04-13 093651](https://github.com/user-attachments/assets/5a58f287-ae11-4bde-8ee7-fe2470cf9448)


## ✍️ Author
- **Shubham** – [LinkedIn](https://linkedin.com/newbieshubham) | [Email](mailto:shubham30p@gmail.com)

## 📄 License
This project is licensed under the MIT License.

---

Feel free to fork, contribute or give feedback!
