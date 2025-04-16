# Link to Google colab notebook(ipynb) file : https://colab.research.google.com/drive/14xttmQZ5f6i4EyBETyt_i9kV0ryF0g3e?usp=sharing

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def clean_and_engineer_features(csv_path):
    # Load the dataset
    df = pd.read_csv(csv_path)

    # Drop 'Unnamed: 0' if it exists
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    # Convert 'Date' to datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Create new features
    df['price_change'] = df['Close'] - df['Open']
    df['high_low_spread'] = df['High'] - df['Low']
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['month'] = df['Date'].dt.month

    # Drop 'Date' if not needed
    df = df.drop(columns=['Date'])

    # Separate features and target
    features = df.drop(columns=['Close'])
    target = df['Close']

    # Feature scaling
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Combine scaled features with target into a new DataFrame
    scaled_df = pd.DataFrame(scaled_features, columns=features.columns)
    scaled_df['Close'] = target.values

    return scaled_df


if __name__ == "__main__":
    input_csv = "goldstock.csv"  # Change this path as needed
    output_df = clean_and_engineer_features(input_csv)
    output_df.to_csv("goldstock_final.csv", index=False)
