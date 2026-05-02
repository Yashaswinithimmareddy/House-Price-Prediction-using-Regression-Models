import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

def preprocess_data(data_path, model_dir):
    """Loads data, performs feature engineering, scaling, encoding and splits."""
    print("Loading and preprocessing data...")
    df = pd.read_csv(data_path)
    
    # Feature Engineering Example: Total Rooms
    df['Total_Rooms'] = df['Bedrooms'] + df['Bathrooms']
    
    # Define features and target
    X = df.drop('Price', axis=1)
    y = df['Price']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Identify numerical and categorical columns
    numeric_features = ['Area_sqft', 'Bedrooms', 'Bathrooms', 'Age_Years', 'Total_Rooms']
    categorical_features = ['Location', 'Furnishing', 'Parking']
    
    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Fit and transform the data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Save the preprocessor
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(preprocessor, os.path.join(model_dir, 'preprocessor.pkl'))
    
    print("Preprocessing completed. Preprocessor saved.")
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor

if __name__ == "__main__":
    preprocess_data('../data/housing_data.csv', '../models')
