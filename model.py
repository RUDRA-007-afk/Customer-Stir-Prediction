# Customer Churn Prediction - Complete Machine Learning Pipeline
# =============================================================

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report, 
                           roc_auc_score, roc_curve)
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("🚀 Customer Churn Prediction Pipeline")
print("=" * 50)

# 1. CREATE SAMPLE DATASET (Replace this with your actual data loading)
# ====================================================================
print("\n1. Creating Sample Dataset...")

# Generate synthetic customer data
np.random.seed(42)
n_customers = 5000

# Create sample data
data = {
    'customer_id': range(1, n_customers + 1),
    'age': np.random.normal(45, 15, n_customers).astype(int),
    'gender': np.random.choice(['Male', 'Female'], n_customers),
    'tenure_months': np.random.poisson(24, n_customers),
    'monthly_charges': np.random.normal(65, 20, n_customers),
    'total_charges': np.random.normal(1500, 800, n_customers),
    'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], 
                                    n_customers, p=[0.5, 0.3, 0.2]),
    'payment_method': np.random.choice(['Electronic check', 'Mailed check', 
                                      'Bank transfer', 'Credit card'], n_customers),
    'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], 
                                       n_customers, p=[0.4, 0.4, 0.2]),
    'online_security': np.random.choice(['Yes', 'No', 'No internet service'], 
                                      n_customers, p=[0.3, 0.5, 0.2]),
    'tech_support': np.random.choice(['Yes', 'No', 'No internet service'], 
                                   n_customers, p=[0.3, 0.5, 0.2]),
    'streaming_tv': np.random.choice(['Yes', 'No', 'No internet service'], 
                                   n_customers, p=[0.4, 0.4, 0.2]),
    'paperless_billing': np.random.choice(['Yes', 'No'], n_customers, p=[0.6, 0.4]),
    'senior_citizen': np.random.choice([0, 1], n_customers, p=[0.8, 0.2])
}

# Create churn based on logical patterns
churn_probability = (
    0.1 +  # Base probability
    0.3 * (data['contract_type'] == 'Month-to-month').astype(int) +
    0.2 * (data['tenure_months'] < 6) +
    0.15 * (data['monthly_charges'] > 80) +
    0.1 * (data['senior_citizen'] == 1) +
    0.1 * (data['payment_method'] == 'Electronic check').astype(int) -
    0.2 * (data['online_security'] == 'Yes').astype(int) -
    0.15 * (data['tech_support'] == 'Yes').astype(int)
)

# Ensure probabilities are between 0 and 1
churn_probability = np.clip(churn_probability, 0, 1)
data['churn'] = np.random.binomial(1, churn_probability, n_customers)

# Create DataFrame
df = pd.DataFrame(data)

# Clean up the data
df['age'] = np.clip(df['age'], 18, 95)  # Reasonable age range
df['monthly_charges'] = np.clip(df['monthly_charges'], 20, 150)  # Reasonable charges
df['total_charges'] = np.maximum(df['total_charges'], df['monthly_charges'])  # Total >= monthly

print(f"✅ Dataset created with {df.shape[0]} customers and {df.shape[1]} features")

# 2. EXPLORE THE DATASET
# ======================
print("\n2. Dataset Exploration...")

print("\n📊 Dataset Info:")
print(df.info())

print("\n📈 Summary Statistics:")
print(df.describe())

print("\n🎯 Churn Distribution:")
churn_counts = df['churn'].value_counts()
print(f"No Churn: {churn_counts[0]} ({churn_counts[0]/len(df)*100:.1f}%)")
print(f"Churn: {churn_counts[1]} ({churn_counts[1]/len(df)*100:.1f}%)")

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Churn distribution
axes[0,0].pie(churn_counts.values, labels=['No Churn', 'Churn'], autopct='%1.1f%%', 
              colors=['lightblue', 'lightcoral'])
axes[0,0].set_title('Churn Distribution')

# Age distribution by churn
df.boxplot(column='age', by='churn', ax=axes[0,1])
axes[0,1].set_title('Age Distribution by Churn')
axes[0,1].set_xlabel('Churn')

# Monthly charges by churn
df.boxplot(column='monthly_charges', by='churn', ax=axes[1,0])
axes[1,0].set_title('Monthly Charges by Churn')
axes[1,0].set_xlabel('Churn')

# Tenure by churn
df.boxplot(column='tenure_months', by='churn', ax=axes[1,1])
axes[1,1].set_title('Tenure (Months) by Churn')
axes[1,1].set_xlabel('Churn')

plt.tight_layout()
plt.show()

# Categorical features analysis
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Contract type vs churn
contract_churn = pd.crosstab(df['contract_type'], df['churn'], normalize='index')
contract_churn.plot(kind='bar', ax=axes[0,0], color=['lightblue', 'lightcoral'])
axes[0,0].set_title('Churn Rate by Contract Type')
axes[0,0].set_ylabel('Churn Rate')
axes[0,0].legend(['No Churn', 'Churn'])
axes[0,0].tick_params(axis='x', rotation=45)

# Payment method vs churn
payment_churn = pd.crosstab(df['payment_method'], df['churn'], normalize='index')
payment_churn.plot(kind='bar', ax=axes[0,1], color=['lightblue', 'lightcoral'])
axes[0,1].set_title('Churn Rate by Payment Method')
axes[0,1].set_ylabel('Churn Rate')
axes[0,1].legend(['No Churn', 'Churn'])
axes[0,1].tick_params(axis='x', rotation=45)

# Internet service vs churn
internet_churn = pd.crosstab(df['internet_service'], df['churn'], normalize='index')
internet_churn.plot(kind='bar', ax=axes[1,0], color=['lightblue', 'lightcoral'])
axes[1,0].set_title('Churn Rate by Internet Service')
axes[1,0].set_ylabel('Churn Rate')
axes[1,0].legend(['No Churn', 'Churn'])
axes[1,0].tick_params(axis='x', rotation=45)

# Senior citizen vs churn
senior_churn = pd.crosstab(df['senior_citizen'], df['churn'], normalize='index')
senior_churn.plot(kind='bar', ax=axes[1,1], color=['lightblue', 'lightcoral'])
axes[1,1].set_title('Churn Rate by Senior Citizen Status')
axes[1,1].set_ylabel('Churn Rate')
axes[1,1].legend(['No Churn', 'Churn'])
axes[1,1].set_xticks([0, 1])
axes[1,1].set_xticklabels(['No', 'Yes'], rotation=0)

plt.tight_layout()
plt.show()

# 3. DATA PREPROCESSING
# =====================
print("\n3. Data Preprocessing...")

# Check for missing values
print("Missing values:")
print(df.isnull().sum())

# Create a copy for preprocessing
df_processed = df.copy()

# Drop customer_id as it's not useful for prediction
df_processed = df_processed.drop('customer_id', axis=1)

# Identify categorical and numerical columns
categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_cols.remove('churn')  # Remove target variable

print(f"Categorical columns: {categorical_cols}")
print(f"Numerical columns: {numerical_cols}")

# Encode categorical variables
df_encoded = df_processed.copy()

# Use LabelEncoder for binary categorical variables
binary_cols = []
for col in categorical_cols:
    if df_processed[col].nunique() == 2:
        binary_cols.append(col)
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_processed[col])

print(f"Binary encoded columns: {binary_cols}")

# Use OneHotEncoder for multi-class categorical variables
multi_class_cols = [col for col in categorical_cols if col not in binary_cols]
if multi_class_cols:
    df_encoded = pd.get_dummies(df_encoded, columns=multi_class_cols, drop_first=True)
    print(f"One-hot encoded columns: {multi_class_cols}")

print(f"✅ Dataset shape after encoding: {df_encoded.shape}")

# 4. SPLIT THE DATASET
# ====================
print("\n4. Splitting Dataset...")

# Separate features and target
X = df_encoded.drop('churn', axis=1)
y = df_encoded['churn']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Features standardized")

# 5. MODEL BUILDING
# ==================
print("\n5. Building Models...")

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
}

# Train models
trained_models = {}
for name, model in models.items():
    print(f"Training {name}...")
    
    # Use scaled data for Logistic Regression, original for tree-based models
    if name == 'Logistic Regression':
        model.fit(X_train_scaled, y_train)
        trained_models[name] = (model, 'scaled')
    else:
        model.fit(X_train, y_train)
        trained_models[name] = (model, 'original')

print("✅ All models trained")

# 6. MODEL EVALUATION
# ====================
print("\n6. Model Evaluation...")

results = {}

for name, (model, data_type) in trained_models.items():
    print(f"\n📊 {name} Results:")
    print("-" * 30)
    
    # Make predictions
    if data_type == 'scaled':
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Store results
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    # Print metrics
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# 7. MODEL COMPARISON
# ===================
print("\n7. Model Comparison...")

# Create comparison DataFrame
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[model]['accuracy'] for model in results.keys()],
    'Precision': [results[model]['precision'] for model in results.keys()],
    'Recall': [results[model]['recall'] for model in results.keys()],
    'F1-Score': [results[model]['f1'] for model in results.keys()],
    'ROC-AUC': [results[model]['roc_auc'] for model in results.keys()]
})

print("\n📋 Model Performance Comparison:")
print(comparison_df.round(4))

# Find best model based on F1-score (balanced metric for imbalanced data)
best_model_name = comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Model']
print(f"\n🏆 Best Model: {best_model_name} (F1-Score: {comparison_df['F1-Score'].max():.4f})")

# Visualize model comparison
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Bar plot of metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
x = np.arange(len(metrics))
width = 0.25

for i, model in enumerate(results.keys()):
    values = [results[model][metric.lower().replace('-', '_')] for metric in metrics]
    axes[0].bar(x + i*width, values, width, label=model, alpha=0.8)

axes[0].set_xlabel('Metrics')
axes[0].set_ylabel('Score')
axes[0].set_title('Model Performance Comparison')
axes[0].set_xticks(x + width)
axes[0].set_xticklabels(metrics, rotation=45)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# ROC Curves
for name in results.keys():
    fpr, tpr, _ = roc_curve(y_test, results[name]['probabilities'])
    axes[1].plot(fpr, tpr, label=f"{name} (AUC = {results[name]['roc_auc']:.3f})")

axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curves')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, (name, result) in enumerate(results.items()):
    cm = confusion_matrix(y_test, result['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues')
    axes[i].set_title(f'{name}\nConfusion Matrix')
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# 8. FEATURE IMPORTANCE (for tree-based models)
# ==============================================
print("\n8. Feature Importance Analysis...")

# Get the best model
best_model, best_data_type = trained_models[best_model_name]

if best_model_name in ['Decision Tree', 'Random Forest']:
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔍 Top 10 Most Important Features ({best_model_name}):")
    print(feature_importance.head(10))
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(15)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title(f'Top 15 Feature Importance - {best_model_name}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

# 9. SAVE THE BEST MODEL
# =======================
print("\n9. Saving the Best Model...")

# Save the best model and scaler
model_filename = f'best_churn_model_{best_model_name.lower().replace(" ", "_")}.joblib'
scaler_filename = 'feature_scaler.joblib'

joblib.dump(best_model, model_filename)
joblib.dump(scaler, scaler_filename)

print(f"✅ Best model saved as: {model_filename}")
print(f"✅ Feature scaler saved as: {scaler_filename}")

# Create a model info file
model_info = {
    'model_name': best_model_name,
    'model_type': type(best_model).__name__,
    'feature_columns': list(X.columns),
    'performance_metrics': {
        'accuracy': results[best_model_name]['accuracy'],
        'precision': results[best_model_name]['precision'],
        'recall': results[best_model_name]['recall'],
        'f1_score': results[best_model_name]['f1'],
        'roc_auc': results[best_model_name]['roc_auc']
    },
    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
}

# Save model info
import json
with open('model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print("✅ Model information saved as: model_info.json")

# Example of how to load and use the saved model
print("\n10. Example: Loading and Using the Saved Model")
print("=" * 50)

print("""
# To load and use the saved model in the future:

import joblib
import pandas as pd

# Load the model and scaler
model = joblib.load('{}')
scaler = joblib.load('{}')

# Load model info
with open('model_info.json', 'r') as f:
    model_info = json.load(f)

# Make predictions on new data
# new_data = pd.DataFrame(...)  # Your new customer data
# new_data_scaled = scaler.transform(new_data)  # If using scaled model
# predictions = model.predict(new_data_scaled)
# probabilities = model.predict_proba(new_data_scaled)[:, 1]
""".format(model_filename, scaler_filename))

print("\n🎉 Customer Churn Prediction Pipeline Complete!")
print("=" * 50)
print(f"📊 Best Model: {best_model_name}")
print(f"🎯 F1-Score: {results[best_model_name]['f1']:.4f}")
print(f"📈 ROC-AUC: {results[best_model_name]['roc_auc']:.4f}")
print("\n💡 Next Steps:")
print("1. Deploy the model to production")
print("2. Monitor model performance over time")
print("3. Retrain with new data periodically")
print("4. Implement churn prevention strategies for high-risk customers")