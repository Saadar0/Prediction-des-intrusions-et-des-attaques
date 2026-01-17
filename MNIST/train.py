# ============================================
# MNIST Handwritten Digit Recognition
# ============================================

# 1. Import libraries
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import joblib
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================
# 2. Load MNIST dataset
# ============================================
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("Training set:", X_train.shape)
print("Test set:", X_test.shape)

# ============================================
# 3. Preprocessing
# ============================================

# Normalize pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# Flatten images (28x28 → 784)
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# ============================================
# 4. Train the model (multiclass classification)
# ============================================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Model trained successfully!")

# ============================================
# 5. Prediction
# ============================================
y_pred = model.predict(X_test)

# ============================================
# 6. Evaluation
# ============================================

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ============================================
# 7. Visualize Predictions
# ============================================
indices = np.random.choice(len(X_test), 10, replace=False)

plt.figure(figsize=(12, 4))
for i, idx in enumerate(indices):
    image = X_test[idx].reshape(28, 28)
    true_label = y_test[idx]
    predicted_label = y_pred[idx]

    plt.subplot(2, 5, i + 1)
    plt.imshow(image, cmap="gray")
    plt.title(f"True: {true_label} | Pred: {predicted_label}")
    plt.axis("off")

plt.tight_layout()
plt.show()

# Save trained model
joblib.dump(model, "mnist_digit_model.pkl")

print("Model saved successfully!")

# ============================================
# END
# ============================================
