import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ── 1. Load & select easy-to-understand features ──────────────────────────────
df = pd.read_csv('train.csv')

FEATURES = [
    'OverallQual',   # Overall quality (1–10)
    'GrLivArea',     # Above-ground living area (sq ft)
    'GarageCars',    # Garage capacity (cars)
    'TotalBsmtSF',   # Basement area (sq ft)
    'FullBath',      # Full bathrooms
    'YearBuilt',     # Year built
    'BedroomAbvGr',  # Bedrooms above ground
    'LotArea',       # Lot size (sq ft)
]
TARGET = 'SalePrice'

data = df[FEATURES + [TARGET]].dropna()
print(f"Using {len(data)} rows after dropping nulls")
print(data.describe())

# ── 2. EDA plots ──────────────────────────────────────────────────────────────
os.makedirs('plots', exist_ok=True)

# Price distribution
plt.figure(figsize=(8, 4))
sns.histplot(data[TARGET], bins=50, kde=True, color='steelblue')
plt.title('Sale Price Distribution')
plt.xlabel('Sale Price ($)')
plt.tight_layout()
plt.savefig('plots/price_distribution.png', dpi=150)
plt.close()

# Price vs Living Area
plt.figure(figsize=(7, 4))
sns.scatterplot(data=data, x='GrLivArea', y=TARGET, hue='OverallQual', palette='RdYlGn', alpha=0.6)
plt.title('Sale Price vs Living Area (colored by Quality)')
plt.xlabel('Above-ground Living Area (sq ft)')
plt.ylabel('Sale Price ($)')
plt.tight_layout()
plt.savefig('plots/price_vs_area.png', dpi=150)
plt.close()

# Price by Overall Quality
plt.figure(figsize=(8, 4))
sns.boxplot(data=data, x='OverallQual', y=TARGET, palette='RdYlGn')
plt.title('Sale Price by Overall Quality')
plt.xlabel('Overall Quality (1–10)')
plt.ylabel('Sale Price ($)')
plt.tight_layout()
plt.savefig('plots/price_vs_quality.png', dpi=150)
plt.close()

# Price by Bedrooms
plt.figure(figsize=(7, 4))
sns.boxplot(data=data, x='BedroomAbvGr', y=TARGET, palette='Set2')
plt.title('Sale Price by Bedrooms')
plt.xlabel('Bedrooms Above Ground')
plt.ylabel('Sale Price ($)')
plt.tight_layout()
plt.savefig('plots/price_vs_bedrooms.png', dpi=150)
plt.close()

# Year Built vs Price
plt.figure(figsize=(8, 4))
sns.scatterplot(data=data, x='YearBuilt', y=TARGET, alpha=0.4, color='mediumseagreen')
plt.title('Sale Price vs Year Built')
plt.xlabel('Year Built')
plt.ylabel('Sale Price ($)')
plt.tight_layout()
plt.savefig('plots/price_vs_year.png', dpi=150)
plt.close()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('plots/correlation_heatmap.png', dpi=150)
plt.close()

print("EDA plots saved.")

# ── 3. Train / test split ─────────────────────────────────────────────────────
X = data[FEATURES]
y = data[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── 4. Train both models ──────────────────────────────────────────────────────
lr = LinearRegression()
lr.fit(X_train, y_train)

rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# ── 5. Evaluate ───────────────────────────────────────────────────────────────
for name, model in [('Linear Regression', lr), ('Random Forest', rf)]:
    pred = model.predict(X_test)
    print(f"\n{name}")
    print(f"  R²  : {r2_score(y_test, pred):.4f}")
    print(f"  MAE : ${mean_absolute_error(y_test, pred):,.0f}")

# ── 6. Save best model ────────────────────────────────────────────────────────
rf_r2 = r2_score(y_test, rf.predict(X_test))
lr_r2 = r2_score(y_test, lr.predict(X_test))
best = rf if rf_r2 >= lr_r2 else lr
pickle.dump(best, open('model.pkl', 'wb'))
print(f"\nSaved best model (model.pkl)")
