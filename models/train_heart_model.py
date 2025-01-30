import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score  # Import accuracy_score

# ✅ Load the heart disease dataset
data = pd.read_csv("../data/heart.csv")  # Ensure this file exists

# ✅ Print the column names (for debugging)
print("Columns in dataset:", data.columns)

# ✅ Identify the correct target column (change based on actual dataset)
target_column = 'target'  # Change this if your dataset has a different label for the outcome

# ✅ Check if the column exists
if target_column not in data.columns:
    raise KeyError(f"Column '{target_column}' not found in dataset!")

# ✅ Select features (X) and target (y)
X = data.drop(columns=[target_column])  # Drop the target column
y = data[target_column]  # Target variable

# ✅ Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ✅ Save trained model inside `saved_models` folder
pickle.dump(model, open("../saved_models/heart_model.sav", 'wb'))

print("Heart Disease Model Trained and Saved Successfully!")

# Use the trained model to predict on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Heart Model Accuracy:", accuracy)
