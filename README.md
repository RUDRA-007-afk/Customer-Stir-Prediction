# Customer Churn Prediction Model

A machine learning pipeline to predict customer churn using historical customer data and service usage patterns.

## Overview

This project helps businesses identify customers who are likely to stop using their services (churn). By analyzing customer demographics, account information, and service usage patterns, the model can predict which customers are at risk of leaving, enabling proactive retention strategies.

## Key Features

- **Data Processing**: Automated data cleaning, encoding, and feature scaling
- **Multiple ML Models**: Logistic Regression, Decision Tree, and Random Forest
- **Model Evaluation**: Comprehensive performance metrics and visualizations
- **Feature Analysis**: Identifies most important factors affecting churn
- **Production Ready**: Saves trained models for future predictions

## Functionality

### 1. Data Analysis
- Loads and explores customer dataset
- Visualizes churn patterns and customer segments
- Analyzes relationship between features and churn

### 2. Data Preprocessing
- Handles missing values
- Encodes categorical variables
- Scales numerical features
- Splits data into training and testing sets

### 3. Model Training
- Trains three different machine learning models
- Compares model performance
- Automatically selects the best performing model

### 4. Model Evaluation
- Calculates accuracy, precision, recall, and F1-score
- Generates confusion matrices and ROC curves
- Provides detailed performance comparison

### 5. Results & Insights
- Identifies key factors that influence customer churn
- Saves the best model for production use
- Provides actionable insights for business decisions

## Quick Start

1. **Install dependencies:**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

2. **Run the pipeline:**
```bash
python churn_prediction.py
```

3. **Use your own data:**
Replace the sample data section with:
```python
df = pd.read_csv('your_customer_data.csv')
```

## Output Files

- `best_churn_model_*.joblib` - Trained model ready for predictions
- `feature_scaler.joblib` - Feature scaling parameters
- `model_info.json` - Model performance metrics and metadata

## Use Cases

- **Customer Retention**: Identify high-risk customers for targeted retention campaigns
- **Business Intelligence**: Understand factors that drive customer churn
- **Revenue Protection**: Proactively prevent customer losses
- **Marketing Strategy**: Optimize customer acquisition and retention efforts
