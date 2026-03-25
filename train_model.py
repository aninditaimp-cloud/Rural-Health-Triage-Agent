import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

print("1. Loading raw dataset...")
df = pd.read_csv('symptoms.csv')

# Clean the string data (remove trailing spaces)
for col in df.columns:
    df[col] = df[col].str.strip()

print("2. Preprocessing data (One-Hot Encoding)...")
# Get all symptom columns
symptom_cols = [col for col in df.columns if col != 'Disease']

# Transform the text columns into a binary matrix (0s and 1s)
# This creates a dataframe where every unique symptom is its own column
encoded_features = pd.get_dummies(df[symptom_cols].stack()).groupby(level=0).max()
encoded_features['Disease'] = df['Disease']

print(f"   -> Found {len(encoded_features.columns) - 1} unique symptoms.")

X = encoded_features.drop('Disease', axis=1)
y = encoded_features['Disease']

# Save the exact list of feature columns so the main app knows what to expect
feature_names = list(X.columns)
joblib.dump(feature_names, 'model_features.pkl')

print("3. Training the Random Forest Classifier...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"4. Model trained with accuracy: {accuracy * 100:.2f}%")

print("5. Saving model to disk...")
joblib.dump(model, 'triage_model.pkl')
print("   -> Success! 'triage_model.pkl' and 'model_features.pkl' saved.")