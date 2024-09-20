# Gold Stock Price Prediction
This project focuses on predicting the movement of gold stock prices using historical market data. The dataset includes key financial columns such as Open, Close, High, Low, and Volume. Through exploratory data analysis (EDA), visualizations are created using Matplotlib and Seaborn to better understand trends and relationships between these variables.

## Project Features:
## Exploratory Data Analysis (EDA):

Visualized key columns like Open, Close, High, Low, and Volume to identify trends and patterns.
Utilized Matplotlib and Seaborn for generating meaningful visualizations to guide model building.
## Feature Engineering:

Added custom features like open-close, low-high, and a binary indicator is_quarter_end (whether the month is a quarter-end).
## Machine Learning Models:

Implemented Logistic Regression and XGBoost (XGBClassifier) using Scikit-learn for binary classification to predict future price movements.
Training and validation were performed using a train-test split, and performance was evaluated using the ROC-AUC score for each model.
## Workflow:
## Data Preprocessing: Scaled the features using StandardScaler.
## Model Training: Trained multiple machine learning models, including Logistic Regression and XGBClassifier.
## Model Evaluation: Evaluated models using AUC-ROC scores on both training and validation datasets.
## Key Dependencies:
Python 3.x
Pandas
Matplotlib
Seaborn
Scikit-learn
XGBoost
## Future Improvements:
Hyperparameter tuning of models.
Testing additional machine learning algorithms.
Incorporating more features to improve prediction accuracy.
