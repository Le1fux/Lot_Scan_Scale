# data_preprocessing.py
# import cv2
# import numpy as np
# import os

def load_images(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(os.path.join(folder, filename))
            img = cv2.resize(img, (128, 128))
            images.append(img / 255.0)
            label = os.path.splitext(filename)[0]
            labels.append(label)
    return np.array(images), np.array(labels)


if __name__ == "__main__":
    X, y = load_images("../data/sample_labels")
    print(f"Loaded {len(X)} images.")
