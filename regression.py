import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score
from scipy.stats import zscore
import joblib

# -------------------------
# Load Data
# -------------------------
X_train = np.load("X_train.npy")
y_train = np.load("Y_train.npy")
print("Data shapes:", X_train.shape, y_train.shape)

# -------------------------
# Feature Selection
# -------------------------
df = pd.DataFrame(X_train, columns=[f"x{i+1}" for i in range(X_train.shape[1])])
df["y"] = y_train

# Keep features with correlation > 0.1
corr_with_y = df.corr()["y"].drop("y")
selected_features = corr_with_y[abs(corr_with_y) > 0.1].index.tolist()
selected_indices = [int(f[1:])-1 for f in selected_features]

print("Selected features:", selected_features)
X_sel = df[selected_features].values

# -------------------------
# Remove Outliers
# -------------------------
mask = (np.abs(zscore(X_sel)) < 3).all(axis=1)
X_clean = X_sel[mask]
y_clean = y_train[mask]
print(f"Removed {X_sel.shape[0] - X_clean.shape[0]} outliers")

# -------------------------
# Define Pipelines
# -------------------------
pipelines = {
    'poly2_linear': Pipeline([
        ("poly", PolynomialFeatures(2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression())
    ]),
    'poly3_linear': Pipeline([
        ("poly", PolynomialFeatures(3, include_bias=False)),
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression())
    ]),
    'poly2_ridge': Pipeline([
        ("poly", PolynomialFeatures(2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("regressor", Ridge())
    ]),
    'poly2_lasso': Pipeline([
        ("poly", PolynomialFeatures(2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("regressor", Lasso(max_iter=10000))
    ]),
    'rbf_linear': Pipeline([
        ("rbf", RBFSampler(n_components=50, random_state=42)),
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression())
    ]),
    'rbf_ridge': Pipeline([
        ("rbf", RBFSampler(n_components=50, random_state=42)),
        ("scaler", StandardScaler()),
        ("regressor", Ridge())
    ])
}

# -------------------------
# Cross-validation setup
# -------------------------
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# -------------------------
# Helper: Tune alpha for Ridge/Lasso
# -------------------------
def tune_alpha(model, alphas, X, y):
    best_score, best_alpha = -np.inf, None
    for alpha in alphas:
        model.set_params(regressor__alpha=alpha)
        score = cross_val_score(model, X, y, cv=cv, scoring='r2').mean()
        if score > best_score:
            best_score, best_alpha = score, alpha
    model.set_params(regressor__alpha=best_alpha)
    return best_score, best_alpha

# -------------------------
# Test models
# -------------------------
results = {}
for name, model in pipelines.items():
    print(f"\nTesting {name}...")

    if 'ridge' in name:
        score, alpha = tune_alpha(model, [0.01, 0.1, 1, 10], X_clean, y_clean)
        results[name] = {'score': score, 'alpha': alpha}
        print(f"Best alpha={alpha}, CV R2={score:.4f}")
    elif 'lasso' in name:
        score, alpha = tune_alpha(model, [0.001, 0.01, 0.1, 1], X_clean, y_clean)
        results[name] = {'score': score, 'alpha': alpha}
        print(f"Best alpha={alpha}, CV R2={score:.4f}")
    else:
        score = cross_val_score(model, X_clean, y_clean, cv=cv, scoring='r2').mean()
        results[name] = {'score': score}
        print(f"CV R2={score:.4f}")

# -------------------------
# Select best model
# -------------------------
best_model_name = max(results, key=lambda k: results[k]['score'])
best_model = pipelines[best_model_name]
best_model.fit(X_clean, y_clean)

train_r2 = r2_score(y_clean, best_model.predict(X_clean))
print(f"\nBest model: {best_model_name} | CV R2={results[best_model_name]['score']:.4f} | Training R2={train_r2:.4f}")

# -------------------------
# Save model
# -------------------------
model_data = {
    'model': best_model,
    'selected_indices': selected_indices,
    'model_name': best_model_name
}
# joblib.dump(model_data, "regression_model.pkl")
