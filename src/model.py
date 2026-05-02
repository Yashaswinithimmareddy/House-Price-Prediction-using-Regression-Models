from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import numpy as np

def train_and_evaluate(X_train, X_test, y_train, y_test, model_dir, output_dir):
    """Trains regression models, evaluates them, and saves the best one."""
    print("Training models...")
    
    # 1. Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    
    # 2. Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    
    # Evaluation
    def evaluate(y_true, y_pred, model_name):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        result_str = f"--- {model_name} ---\nMAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR2 Score: {r2:.4f}\n"
        print(result_str)
        return result_str, r2
        
    lr_results, lr_r2 = evaluate(y_test, lr_preds, "Linear Regression")
    rf_results, rf_r2 = evaluate(y_test, rf_preds, "Random Forest")
    
    # Save evaluation results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'evaluation_metrics.txt'), 'w') as f:
        f.write("Model Evaluation Results\n")
        f.write("========================\n\n")
        f.write(lr_results)
        f.write("\n")
        f.write(rf_results)
    
    # Save the best model
    best_model = rf_model if rf_r2 > lr_r2 else lr_model
    best_model_name = "Random_Forest" if rf_r2 > lr_r2 else "Linear_Regression"
    
    joblib.dump(best_model, os.path.join(model_dir, 'best_model.pkl'))
    print(f"Best model ({best_model_name}) saved to {model_dir}")
    
    return best_model, rf_preds

if __name__ == "__main__":
    pass # Can't run standalone easily without processed data
