import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess_data(file):
    df = pd.read_csv(file)
    # Example: encode categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = LabelEncoder().fit_transform(df[col])
    # Standardize features
    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])
    return df
