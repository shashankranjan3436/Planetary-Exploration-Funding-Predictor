# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# Load dataset (replace with your actual dataset path)
df = pd.read_excel('your_dataset.xlsx')

# Exploratory Data Analysis (EDA)
# Visualize data distribution
sns.pairplot(df)
plt.show()

# Visualize correlation heatmap
corr_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Feature engineering and data preprocessing steps
# Since data was not much complex so haven't used but could use for complex dataset
# Data preprocessing
# Drop irrelevant columns, handle missing values, and perform one-hot encoding
df = df.drop(['Car_Name'], axis=1)  # Drop irrelevant columns
df = pd.get_dummies(df, drop_first=True)  # One-hot encoding

# Handle missing values if any
df = df.dropna()  # Drop rows with missing values, you can implement more sophisticated strategies

# Split the data into features (X) and target variable (y)
X = df.drop(['Selling_Price'], axis=1)
y = df['Selling_Price']

# Feature scaling using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(dt_regressor, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Best parameters from the grid search
best_params = grid_search.best_params_

# Train the model with the best parameters
# Decision Tree Regressor model
dt_regressor = DecisionTreeRegressor(**best_params)
dt_regressor.fit(X_train, y_train)
# Make predictions
y_pred_dt = dt_regressor.predict(X_test)

# XGBoost Regressor model
xgb_regressor = XGBRegressor()
xgb_regressor.fit(X_train, y_train)
y_pred_xgb = xgb_regressor.predict(X_test)

# Model evaluation
r2_dt = r2_score(y_test, y_pred_dt)
r2_xgb = r2_score(y_test, y_pred_xgb)

# Compare results
print(f"R2 Score - Decision Tree Regressor: {r2_dt}")
print(f"R2 Score - XGBoost Regressor: {r2_xgb}")

# Choose the model with better performance
if r2_dt > r2_xgb:
    print("Decision Tree Regressor performs better.")
else:
    print("XGBoost Regressor performs better.")
