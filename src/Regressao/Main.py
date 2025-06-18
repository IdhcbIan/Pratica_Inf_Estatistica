import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Add parent directory to path to import Lib.py
sys.path.append(os.path.abspath('..'))
from Lib import extract_from_data, compare, compare_linked, plot_param_vs_param

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")

def load_data():
    """Load and prepare the EV data for analysis"""
    df = pd.read_csv('../../Data/EV_Data.csv')
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    print("\nSample data:")
    print(df.head())
    
    print("\nData summary:")
    print(df.describe())
    
    return df

def explore_data(df):
    """Perform exploratory data analysis on the dataset"""
    print("\nUnique regions in the dataset:")
    print(df['region'].unique())
    
    print("\nUnique parameters in the dataset:")
    print(df['parameter'].unique())
    
    print("\nData types:")
    print(df.dtypes)
    
    # Check for missing values
    print("\nMissing values per column:")
    print(df.isnull().sum())

def prepare_regression_data(df, region, param1, param2):
    """
    Prepare data for regression analysis by extracting and formatting
    the specified parameters for a given region
    """
    # Extract data for the specified region and parameters
    data = extract_from_data(df, region, [param1, param2])
    
    # Drop rows with missing values
    data = data.dropna()
    
    print(f"\nPrepared regression data for {region} - {param1} vs {param2}:")
    print(data.head())
    
    return data

def simple_linear_regression(data, x_param, y_param):
    """
    Perform simple linear regression analysis
    
    Parameters:
    data (DataFrame): DataFrame containing the data
    x_param (str): Name of the independent variable column
    y_param (str): Name of the dependent variable column
    
    Returns:
    tuple: (model, X, y, X_train, X_test, y_train, y_test)
    """
    # Prepare X and y
    X = data[[x_param]].values
    y = data[y_param].values
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and fit the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Print model parameters
    print("\nSimple Linear Regression Results:")
    print(f"Intercept (β₀): {model.intercept_:.4f}")
    print(f"Coefficient (β₁): {model.coef_[0]:.4f}")
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    # Calculate adjusted R-squared
    n_train = len(y_train)
    p = 1  # Number of predictors (excluding intercept)
    adj_r2_train = 1 - ((1 - r2_train) * (n_train - 1) / (n_train - p - 1))
    
    n_test = len(y_test)
    adj_r2_test = 1 - ((1 - r2_test) * (n_test - 1) / (n_test - p - 1))
    
    print("\nModel Performance:")
    print(f"R² (training): {r2_train:.4f}")
    print(f"Adjusted R² (training): {adj_r2_train:.4f}")
    print(f"RMSE (training): {rmse_train:.4f}")
    print(f"R² (test): {r2_test:.4f}")
    print(f"Adjusted R² (test): {adj_r2_test:.4f}")
    print(f"RMSE (test): {rmse_test:.4f}")
    
    return model, X, y, X_train, X_test, y_train, y_test

def plot_regression_results(model, X, y, X_train, X_test, y_train, y_test, x_param, y_param):
    """Plot regression results with actual vs predicted values and residuals"""
    # Create predictions for plotting
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Scatter plot with regression line
    ax1.scatter(X_train, y_train, color='blue', alpha=0.6, label='Training data')
    ax1.scatter(X_test, y_test, color='green', alpha=0.6, label='Test data')
    
    # Create a range of X values for the regression line
    X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_range_pred = model.predict(X_range)
    
    ax1.plot(X_range, y_range_pred, color='red', linewidth=2, label='Regression line')
    ax1.set_xlabel(x_param)
    ax1.set_ylabel(y_param)
    ax1.set_title('Linear Regression Model')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: Residuals
    residuals_train = y_train - y_pred_train
    residuals_test = y_test - y_pred_test
    
    ax2.scatter(y_pred_train, residuals_train, color='blue', alpha=0.6, label='Training residuals')
    ax2.scatter(y_pred_test, residuals_test, color='green', alpha=0.6, label='Test residuals')
    ax2.axhline(y=0, color='red', linestyle='--')
    ax2.set_xlabel('Predicted values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residual Plot')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
    # Additional diagnostic plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # QQ plot of residuals
    sm.qqplot(np.concatenate([residuals_train, residuals_test]), line='45', ax=ax1)
    ax1.set_title('QQ Plot of Residuals')
    
    # Histogram of residuals
    ax2.hist(np.concatenate([residuals_train, residuals_test]), bins=20, alpha=0.7)
    ax2.set_xlabel('Residual value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Histogram of Residuals')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

def polynomial_regression(data, x_param, y_param, degree=2):
    """
    Perform polynomial regression analysis
    
    Parameters:
    data (DataFrame): DataFrame containing the data
    x_param (str): Name of the independent variable column
    y_param (str): Name of the dependent variable column
    degree (int): Degree of the polynomial
    
    Returns:
    tuple: (model, X_poly, y, X_train_poly, X_test_poly, y_train, y_test, poly)
    """
    # Prepare X and y
    X = data[[x_param]].values
    y = data[y_param].values
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Transform training and test data
    X_train_poly = poly.transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    # Create and fit the model
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    # Print model parameters
    print(f"\nPolynomial Regression Results (degree={degree}):")
    print(f"Intercept (β₀): {model.intercept_:.4f}")
    for i, coef in enumerate(model.coef_[1:], 1):
        print(f"Coefficient β{i}: {coef:.4f}")
    
    # Make predictions
    y_pred_train = model.predict(X_train_poly)
    y_pred_test = model.predict(X_test_poly)
    
    # Calculate metrics
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    # Calculate adjusted R-squared
    n_train = len(y_train)
    p = degree  # Number of predictors (excluding intercept)
    adj_r2_train = 1 - ((1 - r2_train) * (n_train - 1) / (n_train - p - 1))
    
    n_test = len(y_test)
    adj_r2_test = 1 - ((1 - r2_test) * (n_test - 1) / (n_test - p - 1))
    
    print("\nModel Performance:")
    print(f"R² (training): {r2_train:.4f}")
    print(f"Adjusted R² (training): {adj_r2_train:.4f}")
    print(f"RMSE (training): {rmse_train:.4f}")
    print(f"R² (test): {r2_test:.4f}")
    print(f"Adjusted R² (test): {adj_r2_test:.4f}")
    print(f"RMSE (test): {rmse_test:.4f}")
    
    return model, X_poly, y, X_train_poly, X_test_poly, y_train, y_test, poly

def plot_polynomial_results(model, X, y, X_train, X_test, y_train, y_test, 
                           x_param, y_param, poly, original_X_train, original_X_test):
    """Plot polynomial regression results with actual vs predicted values and residuals"""
    # Create predictions for plotting
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Scatter plot with regression curve
    ax1.scatter(original_X_train, y_train, color='blue', alpha=0.6, label='Training data')
    ax1.scatter(original_X_test, y_test, color='green', alpha=0.6, label='Test data')
    
    # Create a range of X values for the regression curve
    X_range = np.linspace(np.min(original_X_train), np.max(original_X_test), 100).reshape(-1, 1)
    X_range_poly = poly.transform(X_range)
    y_range_pred = model.predict(X_range_poly)
    
    ax1.plot(X_range, y_range_pred, color='red', linewidth=2, label='Regression curve')
    ax1.set_xlabel(x_param)
    ax1.set_ylabel(y_param)
    ax1.set_title('Polynomial Regression Model')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: Residuals
    residuals_train = y_train - y_pred_train
    residuals_test = y_test - y_pred_test
    
    ax2.scatter(y_pred_train, residuals_train, color='blue', alpha=0.6, label='Training residuals')
    ax2.scatter(y_pred_test, residuals_test, color='green', alpha=0.6, label='Test residuals')
    ax2.axhline(y=0, color='red', linestyle='--')
    ax2.set_xlabel('Predicted values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residual Plot')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

def statsmodels_regression(data, x_param, y_param):
    """
    Perform regression analysis using statsmodels for detailed statistics
    
    Parameters:
    data (DataFrame): DataFrame containing the data
    x_param (str): Name of the independent variable column
    y_param (str): Name of the dependent variable column
    """
    # Create formula for regression
    formula = f"{y_param} ~ {x_param}"
    
    # Fit the model
    model = ols(formula, data=data).fit()
    
    # Print summary
    print("\nStatsmodels Regression Results:")
    print(model.summary())
    
    return model

def main():
    """Main function to run the analysis"""
    print("=" * 80)
    print("Linear Regression Analysis for EV Data")
    print("=" * 80)
    
    # Load data
    df = load_data()
    
    # Explore data
    explore_data(df)
    
    # Parameters to analyze
    region = 'Europe'  # You can change this to analyze different regions
    param1 = 'EV sales'
    param2 = 'Electricity demand'
    
    # Compare parameters using existing functions
    print(f"\nComparing {param1} and {param2} in {region}:")
    compare(df, region, param1, param2)
    compare_linked(df, region, param1, param2)
    plot_param_vs_param(df, region, param1, param2, annotate_years=True)
    
    # Prepare data for regression
    regression_data = prepare_regression_data(df, region, param1, param2)
    
    # Simple linear regression
    model, X, y, X_train, X_test, y_train, y_test = simple_linear_regression(
        regression_data, param1, param2
    )
    
    # Plot regression results
    plot_regression_results(model, X, y, X_train, X_test, y_train, y_test, param1, param2)
    
    # Polynomial regression
    poly_model, X_poly, y_poly, X_train_poly, X_test_poly, y_train_poly, y_test_poly, poly = polynomial_regression(
        regression_data, param1, param2, degree=2
    )
    
    # Plot polynomial regression results
    plot_polynomial_results(
        poly_model, X_poly, y_poly, X_train_poly, X_test_poly, 
        y_train_poly, y_test_poly, param1, param2, poly, X_train, X_test
    )
    
    # Detailed statistics using statsmodels
    sm_model = statsmodels_regression(regression_data, param1, param2)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
