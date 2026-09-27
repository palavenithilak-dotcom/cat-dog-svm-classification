import cv2
import joblib
import sys
from skimage.feature import hog

MODEL_FILE = "cat_dog_svm_model.pkl"
IMAGE_SIZE = (64, 64)

if len(sys.argv) < 2:
    print("Usage: python predict.py path/to/image.jpg")
    raise SystemExit

image_path = sys.argv[1]

model = joblib.load(MODEL_FILE)

image = cv2.imread(image_path)

if image is None:
    print("Could not read the image.")
    raise SystemExit

image = cv2.resize(image, IMAGE_SIZE)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

features = hog(
    gray,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys"
)

prediction = model.predict([features])[0]

if prediction == 0:
    print("Prediction: CAT")
else:
    print("Prediction: DOG")