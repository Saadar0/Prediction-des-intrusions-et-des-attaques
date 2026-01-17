import numpy as np
import joblib
from PIL import Image
import os
import matplotlib.pyplot as plt


def predict_digit():
    # 1. Load the saved model
    try:
        model = joblib.load("mnist_digit_model.pkl")
    except FileNotFoundError:
        print("Error: 'mnist_digit_model.pkl' not found. Run train.py first.")
        return

    # 2. Automatically find an image in the current directory
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    image_files = [f for f in os.listdir('.') if f.lower().endswith(valid_extensions)]

    if not image_files:
        print("No image files found in the directory!")
        return

    # Use the first valid image found
    image_path = "7.png"
    print(f"Processing image: {image_path}")

    try:
        # 3. Preprocess the image to match MNIST standards
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img = img.resize((28, 28))  # Resize to 28x28

        img_array = np.array(img)

        # MNIST uses black backgrounds (0) and white digits (255)
        # Invert colors if the background is light
        if img_array[0, 0] > 127:
            img_array = 255 - img_array

        # Normalize pixel values
        img_array_normalized = img_array / 255.0

        # Flatten image to 1D array of 784 features
        img_flattened = img_array_normalized.reshape(1, -1)

        # 4. Predict using the trained Logistic Regression model
        prediction = model.predict(img_flattened)
        probability = model.predict_proba(img_flattened)

        result_text = f"Predicted: {prediction[0]} ({np.max(probability) * 100:.2f}%)"
        print("-" * 30)
        print(f"RESULT FOR {image_path}")
        print(result_text)
        print("-" * 30)

        # 5. Visualisation
        plt.figure(figsize=(5, 5))
        plt.imshow(img_array, cmap="gray")
        plt.title(result_text)
        plt.axis("off")
        plt.show()

    except Exception as e:
        print(f"An error occurred while processing the image: {e}")


if __name__ == "__main__":
    predict_digit()