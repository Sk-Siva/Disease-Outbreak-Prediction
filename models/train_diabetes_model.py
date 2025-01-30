import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score  # Import accuracy_score

# ✅ Load dataset correctly
data = pd.read_csv("../data/diabetes.csv")  # Ensure this file exists

# ✅ Fix feature & target selection
X = data.drop(columns=['Outcome'])  # Drop the target column
y = data['Outcome']  # Target variable

# ✅ Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ✅ Save trained model inside `saved_models` folder
pickle.dump(model, open("../saved_models/diabetes_model.sav", 'wb'))

print("Diabetes Model Trained and Saved Successfully!")

# Use the trained model to predict on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Diabetes Model Accuracy:", accuracy)
