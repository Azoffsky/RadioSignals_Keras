# USAGE

# python train_vgg.py --dataset signals --model output\vgg_model.model --label-bin output\vgg_lb.pickle --plot output\vgg_plot.png

import matplotlib
matplotlib.use("Agg")

from vgg_like_network import VGG_like_network
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import plot_model
from imutils import paths
import argparse
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import os
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True)
ap.add_argument("-m", "--model", required=True)
ap.add_argument("-l", "--label-bin", required=True)
ap.add_argument("-p", "--plot", required=True)
args = vars(ap.parse_args())

print("[INFO] loading images...")
data = []
labels = []

imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (64, 64))
    data.append(image)
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

data = np.array(data, dtype='float') / 255.0
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

model = VGG_like_network.create_network(width=64, height=64, depth=3, classes=len(lb.classes_))
plot_model(model, to_file="output\\vgg_like_network.png", show_shapes=True)

INIT_LR = 1e-3
EPOCHS = 12
BS = 64

print("[INFO] training network...")
opt = RMSprop(learning_rate=INIT_LR, decay=INIT_LR/EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# H = model.fit(aug.flow(trainX, trainY, batch_size=BS),
#               validation_data=(testX, testY),
#               steps_per_epoch=len(trainX)//BS,
#               epochs=EPOCHS)

H = model.fit(trainX, trainY, batch_size=BS,
              validation_data=(testX, testY),
              steps_per_epoch=len(trainX)//BS,
              epochs=EPOCHS)

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=lb.classes_))

N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Network Loss and Accuracy")
plt.xlabel("# Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["plot"])

print("[INFO] serializing network and label binarizer...")
model.save(args["model"])
f = open(args["label_bin"], "wb")
f.write(pickle.dumps(lb))
f.close()
