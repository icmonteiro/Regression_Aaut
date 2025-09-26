import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, RobustScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectKBest, f_regression

# -------------------------
# Load training data
# -------------------------
X_train = np.load("X_train.npy")
y_train = np.load("Y_train.npy")
print(f"Data shapes: {X_train.shape}, {y_train.shape}")
print(f"Target mean: {y_train.mean():.4f}, std: {y_train.std():.4f}")

# -------------------------
# Cross-validation setup
# -------------------------
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# -------------------------
# Pipeline 
# -------------------------
pipeline = Pipeline([
    ("scaler", RobustScaler()),
    ("feature_selector", SelectKBest(f_regression)),
    ("features", "passthrough"),    
    ("regressor", "passthrough")   
])

# -------------------------
# Parameter grid for GridSearch
# -------------------------
param_grid = [
    # Polynomial features - Ridge
    {
        "features": [PolynomialFeatures(include_bias=False)],
        "features__degree": [2, 3, 4],
        "feature_selector__k": [4, 5, 6],
        "regressor": [Ridge()],
        "regressor__alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    },
    # Polynomial features - Lasso
    {
        "features": [PolynomialFeatures(include_bias=False)],
        "features__degree": [2, 3],
        "feature_selector__k": [4, 5, 6],
        "regressor": [Lasso(max_iter=50000)],
        "regressor__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0]
    },
    # Polynomial features - LinearRegression
    {
        "features": [PolynomialFeatures(include_bias=False)],
        "features__degree": [2, 3, 4],
        "feature_selector__k": [4, 5, 6],
        "regressor": [LinearRegression()]
    },
    # RBF features - Ridge
    {
        "features": [RBFSampler(random_state=42)],
        "features__gamma": [0.01, 0.05, 0.1, 0.2, 0.5],
        "features__n_components": [100, 200, 300],
        "feature_selector__k": [4, 5, 6],
        "regressor": [Ridge()],
        "regressor__alpha": [0.01, 0.1, 1.0, 10.0]
    },
    # RBF features - Lasso
    {
        "features": [RBFSampler(random_state=42)],
        "features__gamma": [0.01, 0.05, 0.1, 0.2, 0.5],
        "features__n_components": [100, 200, 300],
        "feature_selector__k": [4, 5, 6],
        "regressor": [Lasso(max_iter=50000)],
        "regressor__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0]
    },
    # RBF features - LinearRegression
    {
        "features": [RBFSampler(random_state=42)],
        "features__gamma": [0.01, 0.05, 0.1, 0.2],
        "features__n_components": [100, 200, 300],
        "feature_selector__k": [4, 5, 6],
        "regressor": [LinearRegression()]
    }
]

# -------------------------
# Grid search
# -------------------------
search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="r2",
    cv=cv,
    n_jobs=-1,
    verbose=1,
    refit=True
)

print("Starting grid search...")
search.fit(X_train, y_train)

# -------------------------
# Best model & evaluation
# -------------------------
best_model = search.best_estimator_
cv_r2 = search.best_score_
train_r2 = r2_score(y_train, best_model.predict(X_train))

print("\n=== Best Model Results ===")
print(f"Best params: {search.best_params_}")
print(f"Cross-validation R² (mean): {cv_r2:.6f}")
print(f"Training R²: {train_r2:.6f}")

# -------------------------
# Save model
# -------------------------
model_data = {
    'model': best_model,
    'model_name': 'enhanced_pipeline',
    'best_params': search.best_params_,
    'cv_score': cv_r2,
    'train_score': train_r2,
    'expected_test_performance': cv_r2
}

joblib.dump(model_data, "regression_model.pkl")
print("\nSaved model to regression_model.pkl")
