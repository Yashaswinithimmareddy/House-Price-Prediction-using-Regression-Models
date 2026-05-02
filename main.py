import os
from src.data_generator import generate_data
from src.preprocessing import preprocess_data
from src.model import train_and_evaluate
from src.visualization import generate_visualizations

def main():
    print("Starting House Price Prediction Pipeline...\n")
    
    # Define paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'housing_data.csv')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
    IMAGES_DIR = os.path.join(BASE_DIR, 'images')
    
    # 1. Generate Synthetic Data
    generate_data(DATA_PATH, num_samples=1500)
    
    # 2. Preprocess Data
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(DATA_PATH, MODELS_DIR)
    
    # 3. Train and Evaluate Models
    best_model, y_pred = train_and_evaluate(X_train, X_test, y_train, y_test, MODELS_DIR, OUTPUTS_DIR)
    
    # 4. Generate Visualizations
    generate_visualizations(DATA_PATH, y_test, y_pred, IMAGES_DIR, best_model)
    
    # 5. Simulate a Prediction
    print("\n--- Virtual Simulation: Predicting a New House Price ---")
    import pandas as pd
    new_house = pd.DataFrame([{
        'Area_sqft': 2500,
        'Bedrooms': 3,
        'Bathrooms': 2,
        'Age_Years': 5,
        'Location': 'Suburbs',
        'Furnishing': 'Semi-Furnished',
        'Parking': 'Yes'
    }])
    new_house['Total_Rooms'] = new_house['Bedrooms'] + new_house['Bathrooms']
    
    # Transform
    new_house_processed = preprocessor.transform(new_house)
    
    # Predict
    predicted_price = best_model.predict(new_house_processed)[0]
    
    sim_output = f"Input Property Details:\n{new_house.iloc[0].to_string()}\n\nPredicted Price: ${predicted_price:,.2f}"
    print(sim_output)
    
    with open(os.path.join(OUTPUTS_DIR, 'sample_prediction.txt'), 'w') as f:
        f.write(sim_output)
        
    print("\nPipeline completed successfully! All outputs saved to their respective directories.")

if __name__ == "__main__":
    main()
