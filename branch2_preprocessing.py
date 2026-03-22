# branch: preprocessing-cleaning

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv("insurance_data_linear.csv")

# Handle missing values (if any) – simple strategy
df = df.dropna()

# Features and target
X = df.drop("charges", axis=1)
y = df["charges"]

# Identify column types
numeric_features = ["age", "bmi", "children"]
categorical_features = ["sex", "smoker", "region"]

# Preprocessor: one‑hot for categoricals, passthrough numerics
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first"), categorical_features),
        ("num", "passthrough", numeric_features),
    ]
)

# Create base pipeline (no training yet in this branch, just transform)
model = LinearRegression()
pipe = Pipeline(steps=[("preprocessor", preprocessor),
                      ("model", model)])

# Simple train-test split, to verify pipeline runs
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipe.fit(X_train, y_train)
print("Pipeline fitted successfully.")