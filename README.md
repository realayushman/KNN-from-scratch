# Custom K-Nearest Neighbors (KNN) Classifier

This is a simple implementation of the K-Nearest Neighbors (KNN) algorithm from scratch using Python and NumPy, without using any built-in ML models. It's designed for learning purposes and demonstrates how KNN works under the hood.

## 📌 Features

- Implements basic KNN logic from scratch
- Uses Euclidean distance to find nearest neighbors
- Predicts the class based on majority voting
- Includes accuracy scoring function

## 🧠 How It Works

1. **Distance Calculation**: Uses Euclidean distance to measure how close test data points are to training points.
2. **Neighbor Selection**: Selects the `k` closest data points.
3. **Majority Voting**: The most common label among the neighbors is chosen as the prediction.

## 🚀 Getting Started

### Prerequisites
Make sure you have the following libraries installed:
```bash
pip install numpy scikit-learn



import numpy as np
from knn import KNN  # assuming your class is saved in knn.py

# Sample training data
x_train = np.array([[1, 2], [2, 3], [3, 4], [6, 7]])
y_train = np.array([0, 0, 1, 1])

# Test data
x_test = np.array([[2, 2], [5, 5]])

# Initialize and train the model
model = KNN(k=3)
model.fit(x_train, y_train)

# Make predictions
y_pred = model.predict(x_test)
print("Predictions:", y_pred)


#Accuracy Scoring
model.score(y_test, y_pred)
