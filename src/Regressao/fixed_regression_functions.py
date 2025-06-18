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

def extract_from_data(df, region, parameters):
    """
    Extract data for a specific region and parameters from the dataset
    
    Parameters:
    df (DataFrame): The main dataset
    region (str): Region to filter by
    parameters (list): List of parameter names to extract
    
    Returns:
    DataFrame: Pivoted data with parameters as columns
    """
    # Filter by region
    region_data = df[df['region'] == region]
    
    # Filter by parameters
    param_data = region_data[region_data['parameter'].isin(parameters)]
    
    # Pivot to get parameters as columns
    pivoted = param_data.pivot(index='year', columns='parameter', values='value')
    
    return pivoted

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

def safe_adjusted_r2(r2, n, p):
    """
    Calculate adjusted R² with safety checks to prevent division by zero
    
    Parameters:
    r2 (float): R-squared value
    n (int): Number of samples
    p (int): Number of features (excluding intercept)
    
    Returns:
    float: Adjusted R-squared or np.nan if calculation is not possible
    """
    if n <= p + 1:
        print(f"Warning: Not enough samples ({n}) compared to features ({p}) for adjusted R²")
        return np.nan
    else:
        return 1 - (1 - r2) * (n - 1) / (n - p - 1)

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
    
    # Check if we have enough data
    if len(X) < 5:  # Minimum data needed for meaningful train/test split
        print(f"Warning: Not enough data points ({len(X)}) for regression analysis")
        return None, X, y, None, None, None, None
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Check if split resulted in empty sets
    if len(X_train) == 0 or len(X_test) == 0:
        print("Warning: Train/test split resulted in empty sets")
        return None, X, y, X_train, X_test, y_train, y_test
    
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
    
    # Calculate test R² only if test set is not empty
    r2_test = r2_score(y_test, y_pred_test) if len(y_test) > 0 else np.nan
    
    # Calculate adjusted R² using safe function
    n_train = len(y_train)
    n_test = len(y_test)
    p = 1  # Number of features (excluding intercept)
    
    adj_r2_train = safe_adjusted_r2(r2_train, n_train, p)
    adj_r2_test = safe_adjusted_r2(r2_test, n_test, p)
    
    # Calculate RMSE
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test)) if len(y_test) > 0 else np.nan
    
    # Print metrics
    print(f"R² (train): {r2_train:.4f}")
    print(f"Adjusted R² (train): {adj_r2_train:.4f}" if not np.isnan(adj_r2_train) else "Adjusted R² (train): Not available")
    print(f"RMSE (train): {rmse_train:.4f}")
    print(f"R² (test): {r2_test:.4f}" if not np.isnan(r2_test) else "R² (test): Not available")
    print(f"Adjusted R² (test): {adj_r2_test:.4f}" if not np.isnan(adj_r2_test) else "Adjusted R² (test): Not available")
    print(f"RMSE (test): {rmse_test:.4f}" if not np.isnan(rmse_test) else "RMSE (test): Not available")
    
    return model, X, y, X_train, X_test, y_train, y_test

def plot_regression_results(model, X, y, X_train, X_test, y_train, y_test, x_param, y_param):
    """Plot regression results with actual vs predicted values and residuals"""
    # Check if model is valid
    if model is None:
        print("Cannot plot results: Invalid model")
        return
        
    # Check if test sets are valid
    if X_test is None or y_test is None or len(X_test) == 0 or len(y_test) == 0:
        print("Warning: Test set is empty, plotting only training data")
        has_test_data = False
    else:
        has_test_data = True
    
    # Create predictions for plotting
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test) if has_test_data else None
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Actual vs Predicted
    ax1.scatter(X_train, y_train, color='blue', alpha=0.6, label='Training data')
    if has_test_data:
        ax1.scatter(X_test, y_test, color='green', alpha=0.6, label='Test data')
    
    # Plot regression line
    x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_range_pred = model.predict(x_range)
    ax1.plot(x_range, y_range_pred, color='red', linewidth=2, label='Regression line')
    
    ax1.set_xlabel(x_param)
    ax1.set_ylabel(y_param)
    ax1.set_title('Linear Regression Model')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: Residuals
    residuals_train = y_train - y_pred_train
    residuals_test = y_test - y_pred_test if has_test_data else None
    
    ax2.scatter(y_pred_train, residuals_train, color='blue', alpha=0.6, label='Training residuals')
    if has_test_data:
        ax2.scatter(y_pred_test, residuals_test, color='green', alpha=0.6, label='Test residuals')
    
    ax2.axhline(y=0, color='red', linestyle='-', linewidth=2)
    ax2.set_xlabel('Predicted values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residual Plot')
    ax2.legend()
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
    
    # Check if we have enough data
    if len(X) < degree + 3:  # Need more samples than polynomial degree + intercept + test
        print(f"Warning: Not enough data points ({len(X)}) for polynomial regression of degree {degree}")
        return None, None, y, None, None, None, None, None
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_poly, y, test_size=0.2, random_state=42
    )
    
    # Check if split resulted in empty sets
    if len(X_train) == 0 or len(X_test) == 0:
        print("Warning: Train/test split resulted in empty sets")
        return None, X_poly, y, X_train, X_test, y_train, y_test, poly
    
    # Create and fit the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Print model parameters
    print(f"\nPolynomial Regression Results (degree={degree}):")
    print(f"Intercept (β₀): {model.intercept_:.4f}")
    for i, coef in enumerate(model.coef_, 1):
        print(f"Coefficient β{i}: {coef:.4f}")
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test) if len(y_test) > 0 else np.nan
    
    # Calculate adjusted R² using safe function
    n_train = len(y_train)
    n_test = len(y_test)
    p = degree  # Number of features (excluding intercept)
    
    adj_r2_train = safe_adjusted_r2(r2_train, n_train, p)
    adj_r2_test = safe_adjusted_r2(r2_test, n_test, p)
    
    # Calculate RMSE
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test)) if len(y_test) > 0 else np.nan
    
    # Print metrics
    print(f"R² (training): {r2_train:.4f}")
    print(f"Adjusted R² (training): {adj_r2_train:.4f}" if not np.isnan(adj_r2_train) else "Adjusted R² (training): Not available")
    print(f"RMSE (training): {rmse_train:.4f}")
    print(f"R² (test): {r2_test:.4f}" if not np.isnan(r2_test) else "R² (test): Not available")
    print(f"Adjusted R² (test): {adj_r2_test:.4f}" if not np.isnan(adj_r2_test) else "Adjusted R² (test): Not available")
    print(f"RMSE (test): {rmse_test:.4f}" if not np.isnan(rmse_test) else "RMSE (test): Not available")
    
    return model, X_poly, y, X_train, X_test, y_train, y_test, poly

def plot_polynomial_results(model, X, y, X_train, X_test, y_train, y_test, 
                           x_param, y_param, poly, X_train_orig, X_test_orig):
    """Plot polynomial regression results"""
    # Check if model is valid
    if model is None:
        print("Cannot plot results: Invalid model")
        return
        
    # Check if test sets are valid
    if X_test is None or y_test is None or len(X_test) == 0 or len(y_test) == 0:
        print("Warning: Test set is empty, plotting only training data")
        has_test_data = False
    else:
        has_test_data = True
    
    # Create figure with one subplot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot data points
    ax1.scatter(X_train_orig, y_train, color='blue', alpha=0.6, label='Training data')
    if has_test_data:
        ax1.scatter(X_test_orig, y_test, color='green', alpha=0.6, label='Test data')
    
    # Plot polynomial curve
    X_range = np.linspace(X_train_orig.min(), X_train_orig.max(), 100).reshape(-1, 1)
    X_range_poly = poly.transform(X_range)
    y_range_pred = model.predict(X_range_poly)
    
    ax1.plot(X_range, y_range_pred, color='red', linewidth=2, label='Regression curve')
    ax1.set_xlabel(x_param)
    ax1.set_ylabel(y_param)
    ax1.set_title('Polynomial Regression Model')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

def statsmodels_regression(data, x_param, y_param):
    """
    Perform regression analysis using statsmodels
    
    Parameters:
    data (DataFrame): DataFrame containing the data
    x_param (str): Name of the independent variable column
    y_param (str): Name of the dependent variable column
    """
    # Check if we have enough data
    if len(data) < 3:  # Minimum data needed for regression
        print(f"Warning: Not enough data points ({len(data)}) for statsmodels regression")
        return None
        
    try:
        # Create formula for regression
        formula = f"{y_param} ~ {x_param}"
        
        # Fit the model
        model = ols(formula, data=data).fit()
        
        # Print summary
        print("\nStatsmodels Regression Results:")
        print(model.summary())
        
        return model
    except Exception as e:
        print(f"Error in statsmodels regression: {str(e)}")
        return None
