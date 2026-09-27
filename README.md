# Cat vs Dog Image Classification using SVM

## Objective
Implement a Support Vector Machine (SVM) to classify images as **cats** or **dogs**.

## Project structure

```text
cat-dog-svm/
│
├── cat_dog_svm.py
├── predict.py
├── requirements.txt
├── README.md
│
└── dataset/
    ├── cats/
    └── dogs/
```

## 1. Install packages

Open the VS Code terminal:

```bash
pip install -r requirements.txt
```

## 2. Add the dataset

Download the Cats vs Dogs dataset from Kaggle and place the images like this:

```text
dataset/
├── cats/
│   ├── cat1.jpg
│   ├── cat2.jpg
│   └── ...
└── dogs/
    ├── dog1.jpg
    ├── dog2.jpg
    └── ...
```

You can use a smaller number of images for a quick demonstration.

## 3. Run the project

```bash
python cat_dog_svm.py
```

The program:
- loads cat and dog images
- resizes them to 64x64
- converts images into numerical features
- splits the data into training and testing sets
- trains a linear SVM
- calculates accuracy
- prints a classification report
- saves the trained model

## 4. Predict a new image

After training:

```bash
python predict.py path/to/your/image.jpg
```

Example:

```bash
python predict.py test_cat.jpg
```

## Important
The ZIP intentionally does not contain the full Kaggle image dataset because it can be very large. Add the downloaded images to `dataset/cats` and `dataset/dogs`.
