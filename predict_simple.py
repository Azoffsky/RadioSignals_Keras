# USAGE

# python predict_simple.py --image images\chirp_55.jpg --model output\simple_model.model --label-bin output\simple_lb.pickle --width 32 --height 32

from tensorflow.keras.models import load_model
import argparse
import pickle
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True)
ap.add_argument("-m", "--model", required=True)
ap.add_argument("-l", "--label-bin", required=True)
ap.add_argument("-w", "--width", type=int, default=28)
ap.add_argument("-e", "--height", type=int, default=28)
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
output = image.copy()
image = cv2.resize(image, (args["width"], args["height"]))
image = image.flatten()
image = image.astype("float") / 255.0
image = image.reshape((1, image.shape[0]))

model = load_model(args["model"])
lb = pickle.loads(open(args["label_bin"], "rb").read())

preds = model.predict(image)
print(preds)

i = preds.argmax(axis=1)[0]
label = lb.classes_[i]

text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

cv2.imshow("Image", output)
cv2.waitKey(0)
