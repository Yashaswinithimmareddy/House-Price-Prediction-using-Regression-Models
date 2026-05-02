import pandas as pd
import numpy as np
import os

def generate_data(output_path, num_samples=1000):
    """Generates synthetic housing data and saves it to a CSV."""
    np.random.seed(42)
    
    # Features
    area = np.random.randint(500, 5000, num_samples) # Square feet
    bedrooms = np.random.randint(1, 6, num_samples)
    bathrooms = np.random.randint(1, 4, num_samples)
    age_of_property = np.random.randint(0, 50, num_samples) # Years
    
    locations = ['Downtown', 'Suburbs', 'Rural', 'City Center']
    location = np.random.choice(locations, num_samples)
    
    furnishing_status = ['Unfurnished', 'Semi-Furnished', 'Fully-Furnished']
    furnishing = np.random.choice(furnishing_status, num_samples)
    
    parking = np.random.choice(['Yes', 'No'], num_samples)
    
    # Base Price calculation (some logical correlation)
    price = (area * 150) + (bedrooms * 50000) + (bathrooms * 30000) - (age_of_property * 2000)
    
    # Add location multiplier
    location_multiplier = {'Downtown': 1.5, 'City Center': 1.8, 'Suburbs': 1.2, 'Rural': 0.8}
    price = price * np.vectorize(location_multiplier.get)(location)
    
    # Add furnishing value
    furnishing_value = {'Unfurnished': 0, 'Semi-Furnished': 20000, 'Fully-Furnished': 50000}
    price = price + np.vectorize(furnishing_value.get)(furnishing)
    
    # Add parking value
    parking_value = {'Yes': 15000, 'No': 0}
    price = price + np.vectorize(parking_value.get)(parking)
    
    # Add some random noise
    noise = np.random.normal(0, 20000, num_samples)
    price = price + noise
    
    # Ensure no negative prices
    price = np.abs(price)
    
    df = pd.DataFrame({
        'Area_sqft': area,
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'Age_Years': age_of_property,
        'Location': location,
        'Furnishing': furnishing,
        'Parking': parking,
        'Price': price
    })
    
    # Round price to nearest whole number
    df['Price'] = df['Price'].round(0)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset generated with {num_samples} records at {output_path}")
    return df

if __name__ == "__main__":
    generate_data('../data/housing_data.csv')
