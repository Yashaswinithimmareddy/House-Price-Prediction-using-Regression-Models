# House Price Prediction using Regression Models 🏠📈

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Regression-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Model-yellow)

## 📌 Project Overview
This project builds a robust **House Price Prediction System** using Machine Learning. It predicts property values based on various features such as area, bedrooms, bathrooms, location, and property age. The project uses a complete end-to-end pipeline covering synthetic data generation, data preprocessing, model training (Linear Regression & Random Forest), evaluation, and visualization.

## 🎯 Problem Statement & Industry Relevance
In the real estate market, properties are often mispriced due to human bias or lack of structured data analysis. This project solves that problem by building an automated valuation model (AVM) similar to those used by **Zillow, banks, and property portals**. It helps buyers make informed decisions, sellers price properties accurately, and banks assess collateral value.

## 🛠 Tech Stack
- **Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn
- **Visualization:** Matplotlib, Seaborn
- **Models Used:** Linear Regression, Random Forest Regressor

## 📊 Dataset
Since real estate data is proprietary, this project uses a synthetically generated realistic dataset containing 1,500 samples.
**Features include:**
- `Area_sqft`, `Bedrooms`, `Bathrooms`, `Age_Years`
- `Location` (Downtown, Suburbs, Rural, City Center)
- `Furnishing` (Unfurnished, Semi-Furnished, Fully-Furnished)
- `Parking` (Yes, No)

## 🏆 Model & Results
Two models were trained and compared:
- **Linear Regression**
- **Random Forest Regressor** (Best Model)

**Evaluation Metrics (Random Forest):**
- **MAE:** ~$34,365
- **RMSE:** ~$43,824
- **R² Score:** ~0.98

## 🚀 How to Run the Project

1. **Clone the repository:**
   ```bash
   git clone <your-repo-link>
   cd House-Price-Prediction-ML
   ```

2. **Create a virtual environment & install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run the full pipeline:**
   ```bash
   python main.py
   ```
   *This will generate data, train the model, save visualizations in `images/`, and output evaluation metrics.*

## 📸 Screenshots & Visualizations
- **Correlation Heatmap:** Located in `images/correlation_heatmap.png`
- **Actual vs Predicted Prices:** Located in `images/actual_vs_predicted.png`
- **Price Distribution:** Located in `images/price_distribution.png`

## 🧠 Learning Outcomes
- Generating synthetic but logical business datasets.
- Feature Engineering and handling categorical variables using `OneHotEncoder`.
- Creating scikit-learn `Pipelines` and `ColumnTransformers`.
- Evaluating regression models using MAE, RMSE, and R².
- Creating professional, business-ready data visualizations.

---
*This project is designed as an industry-oriented portfolio piece for Data Science and Machine Learning roles.*
