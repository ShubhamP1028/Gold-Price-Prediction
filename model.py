import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import precision_score, recall_score, accuracy_score



# Load the preprocessed data
def load_processed_data(filepath):
    return pd.read_csv(filepath)


# Perform EDA
def perform_eda(df):
    print("\n--- Dataset Info ---")
    print(df.info())

    print("\n--- Statistical Summary ---")
    print(df.describe())

    print("\n--- Correlation Matrix ---")
    corr_matrix = df.corr()
    print(corr_matrix)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.show()


# Train Linear Regression Model
def train_model(df, target_column='Close'):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n--- Model Performance ---")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R^2 Score: {r2:.4f}")

    # Convert predictions to integers to simulate classification (for illustrative purposes only)
    y_test_class = np.round(y_test / 100).astype(int)
    y_pred_class = np.round(y_pred / 100).astype(int)

    accuracy = accuracy_score(y_test_class, y_pred_class)
    precision = precision_score(y_test_class, y_pred_class, average='weighted', zero_division=0)
    recall = recall_score(y_test_class, y_pred_class, average='weighted', zero_division=0)

    print("\n--- Classification Metrics (Simulated) ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values[:50], label='Actual', marker='o')
    plt.plot(y_pred[:50], label='Predicted', marker='x')
    plt.title('Actual vs Predicted Close Prices (First 50 Samples)')
    plt.xlabel('Sample Index')
    plt.ylabel('Gold Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return model


# Example usage
if __name__ == "__main__":
    filepath = "scaled_gold_data.csv"  # Update with actual path to scaled dataset
    df = load_processed_data(filepath)
    perform_eda(df)
    model = train_model(df)
