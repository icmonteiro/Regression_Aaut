import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import joblib
from sklearn.linear_model import LinearRegression

X_train = np.load("X_train.npy")
Y_train = np.load("Y_train.npy")

"""
see if some features are irrelevant, redundant, or strongly correlated.
    It ranges from –1 to +1:
        +1 → perfect positive correlation (as one increases, the other increases proportionally).
        –1 → perfect negative correlation (as one increases, the other decreases proportionally).   
        0 → no linear relationship.
"""

print(X_train.shape, Y_train.shape)

# Quick look at correlations
df = pd.DataFrame(X_train, columns=[f"x{i}" for i in range(1, 7)])
df["y"] = Y_train
print(df.corr())


# Get correlations with target y
corr_with_y = df.corr()["y"].drop("y")

# Sort by absolute value
corr_sorted = corr_with_y.abs().sort_values(ascending=False)

print("Correlations with y (sorted by importance):")
print(corr_with_y.loc[corr_sorted.index])

# Polynomial regression pipeline
poly_model = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=1.0))
])

# Cross-validation
poly_scores = cross_val_score(poly_model, X_train, Y_train, cv=5, scoring="r2")
print(f"Polynomial Regression (degree=2) mean R²: {poly_scores.mean():.4f}")

# Fit full model
poly_model.fit(X_train, Y_train)

# Save model
joblib.dump(poly_model, "poly_model.pkl")
print("Polynomial model saved as poly_model.pkl")


from sklearn.linear_model import LinearRegression

# Linear regression pipeline (only scaling)
linear_model = Pipeline([
    ("scaler", StandardScaler()),  # scale features
    ("linreg", LinearRegression())
])

# Cross-validation
linear_scores = cross_val_score(linear_model, X_train, Y_train, cv=5, scoring="r2")
print(f"Linear Regression mean R²: {linear_scores.mean():.4f}")

# Fit full model
linear_model.fit(X_train, Y_train)

# Save model
joblib.dump(linear_model, "linear_model.pkl")
print("Linear model saved as linear_model.pkl")
