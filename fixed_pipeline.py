import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

CSV_FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(CSV_FILE)

# Remove duplicate customer records before splitting
data = df.drop_duplicates(subset=["customerID"]).copy()

# The original dataset has no synthetic leakage columns.
# If present, remove target-derived/post-outcome fields.
data = data.drop(
    columns=["ChurnRiskScore", "ChurnedAlready"],
    errors="ignore"
)

y = data["Churn"].map({"No": 0, "Yes": 1})
X = data.drop(columns=["Churn", "customerID"], errors="ignore").copy()

X["TotalCharges"] = pd.to_numeric(
    X["TotalCharges"], errors="coerce"
)

numeric = X.select_dtypes(include=["number"]).columns.tolist()
categorical = X.select_dtypes(exclude=["number"]).columns.tolist()

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipe, numeric),
    ("categorical", categorical_pipe, categorical)
])

# Split before fitting any preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=2000))
])

model.fit(X_train, y_train)
pred = model.predict(X_test)

metrics = {
    "Accuracy": accuracy_score(y_test, pred),
    "Precision": precision_score(y_test, pred, zero_division=0),
    "Recall": recall_score(y_test, pred, zero_division=0),
    "F1-score": f1_score(y_test, pred, zero_division=0)
}

print("FIXED MODEL RESULTS")
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

print("Confusion matrix:\n", confusion_matrix(y_test, pred))
pd.DataFrame([metrics]).to_csv("fixed_model_metrics.csv", index=False)
