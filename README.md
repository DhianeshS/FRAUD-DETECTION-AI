import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


data = {
    "amount": [100, 5000, 200, 45000, 300, 60000, 150, 70000],
    "location_change": [0, 1, 0, 1, 0, 1, 0, 1],
    "device_change": [0, 1, 0, 1, 0, 1, 0, 1],
    "is_fraud": [0, 1, 0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

def fraud_detection_agent(amount, location_change, device_change):
    input_data = np.array([[amount, location_change, device_change]])
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1] * 100

    if prediction[0] == 1:
        status = "Fraudulent Transaction"
        action = "Block transaction and alert user"
    else:
        status = "Legitimate Transaction"
        action = "Approve transaction"

    return {
        "Amount": amount,
        "Fraud Risk (%)": round(probability, 2),
        "Status": status,
        "Recommended Action": action
    }
result = fraud_detection_agent(
    amount=50000,
    location_change=1,
    device_change=1
)

print("\nAI Fraud Detection Result:")
for key, value in result.items():
    print(f"{key}: {value}")
