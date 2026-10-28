import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

CSV_FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(CSV_FILE)
data = df.copy()

# Deliberate leakage: features directly derived from the target
data["ChurnRiskScore"] = data["Churn"].map({"No": 0.05, "Yes": 0.95})
data["ChurnedAlready"] = data["Churn"].map({"No": 0, "Yes": 1})

# Deliberate leakage: duplicate records can cross the split
data = pd.concat([
    data,
    data.sample(frac=0.15, random_state=42)
], ignore_index=True)

y = data["Churn"].map({"No": 0, "Yes": 1})
X = data.drop(columns=["Churn", "customerID"], errors="ignore")
X = pd.get_dummies(X, drop_first=True, dtype=int)
X = X.apply(pd.to_numeric, errors="coerce")

# Deliberate leakage: fit preprocessing before splitting
imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("LEAKY MODEL RESULTS")
print("Accuracy:", accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred, zero_division=0))
print("Recall:", recall_score(y_test, pred, zero_division=0))
print("F1-score:", f1_score(y_test, pred, zero_division=0))
print("Confusion matrix:\n", confusion_matrix(y_test, pred))
