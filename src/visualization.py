import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def generate_visualizations(data_path, y_test, y_pred, img_dir, model):
    """Generates EDA and result visualizations."""
    print("Generating visualizations...")
    os.makedirs(img_dir, exist_ok=True)
    
    df = pd.read_csv(data_path)
    
    # 1. Price Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Price'], bins=30, kde=True, color='blue')
    plt.title('Distribution of House Prices')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(img_dir, 'price_distribution.png'))
    plt.close()
    
    # 2. Correlation Heatmap (only numeric features)
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(10, 8))
    correlation_matrix = numeric_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'correlation_heatmap.png'))
    plt.close()
    
    # 3. Actual vs Predicted Price (for Random Forest or Best Model)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='green')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title('Actual vs Predicted House Prices')
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'actual_vs_predicted.png'))
    plt.close()
    
    # 4. Feature Importance (if model is RandomForest)
    if hasattr(model, 'feature_importances_'):
        # Just getting a simple representation since exact feature names after OneHotEncoding can be complex to extract simply
        # For simplicity, we just plot top features if possible, or just a generic plot.
        # Since it's a bit complex with pipeline, we'll skip detailed feature names and just plot raw importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[-10:] # top 10
        
        plt.figure(figsize=(10, 6))
        plt.title('Top 10 Feature Importances')
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [f"Feature {i}" for i in indices])
        plt.xlabel('Relative Importance')
        plt.savefig(os.path.join(img_dir, 'feature_importance.png'))
        plt.close()

    print(f"Visualizations saved to {img_dir}")

if __name__ == "__main__":
    pass
