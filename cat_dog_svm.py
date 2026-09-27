import os
import cv2
import numpy as np
import joblib

from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

CAT_FOLDER = "dataset/cats"
DOG_FOLDER = "dataset/dogs"

IMAGE_SIZE = (64, 64)
MAX_IMAGES_PER_CLASS = 1000


def extract_features(image):
    image = cv2.resize(image, IMAGE_SIZE)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Extract HOG features
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    return features


def load_images(folder, label):
    features = []
    labels = []

    files = os.listdir(folder)[:MAX_IMAGES_PER_CLASS]

    for file in files:
        path = os.path.join(folder, file)

        image = cv2.imread(path)

        if image is not None:
            feature = extract_features(image)

            features.append(feature)
            labels.append(label)

    return features, labels


print("Loading cat images...")
cat_features, cat_labels = load_images(CAT_FOLDER, 0)

print("Loading dog images...")
dog_features, dog_labels = load_images(DOG_FOLDER, 1)

X = np.array(cat_features + dog_features)
y = np.array(cat_labels + dog_labels)

print("\nTotal images:", len(X))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training images:", len(X_train))
print("Testing images:", len(X_test))

print("\nTraining SVM model...")

model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale"
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n==============================")
print("SVM MODEL RESULTS")
print("==============================")

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["Cat", "Dog"]
    )
)

joblib.dump(model, "cat_dog_svm_model.pkl")

print("\nModel saved successfully!")
print("File: cat_dog_svm_model.pkl")