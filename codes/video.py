#import h5py
import pickle
import numpy as np
import cv2
import os
from pathlib import Path
import json
import pandas as pd
import keras

################################################################
framewidth = 640
frameheight = 480
brightness = 180
threshold = 0.90
font = cv2.FONT_HERSHEY_SIMPLEX
#################################################################

cwd_path = os.path.abspath('')
print(cwd_path)
parent_path = Path(cwd_path).parent
print(parent_path)

#import json
json_f = open(f"{parent_path}/keys.json", "r")
print(json_f)
parameters = json.load(json_f)
BACKUP_DIR = parameters["backup_dir"]
backup_dir = f"{parent_path }/{BACKUP_DIR}"


# set up the video cam
cap = cv2.VideoCapture(1)
cap.set(3, framewidth)
cap.set(4, frameheight)
cap.set(10, brightness)

# import trained model
h5_in = open(f"{backup_dir}/my_model.h5", "rb")
# model = h5py.File(h5_in)
model = keras.models.load_model(f"{backup_dir}/my_model.h5", compile=False)


def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):
    img = cv2.equalizeHist(img)
    return img


def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img


def getclassname(classNo):
    meta_df = pd.read_csv(f"{parent_path}/image_datasets/Meta.csv")
    for i in meta_df:
        if str(classNo) == str(meta_df[i]["ClassId"]):
            return str(meta_df[i]["label"])


while True:

    # read image data
    success, imgOriginal = cap.read()

    # PROCESS IMAGE
    img = np.asarray(imgOriginal)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    cv2.imshow("Processed Image", img)
    img = img.reshape(1, 32, 32, 1)
    cv2.putText(imgOriginal, "Class: ", (20, 35), font,
                0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOriginal, "Probability: ", (20, 75),
                font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    # # Predict Image
    # predictions = model.predict(img)
    # classIndex = model.predict_classes(img)
    # probability = -np.amax(predictions)

    # if probability > threshold:
    #     cv2.putText(imgOriginal, str(classIndex)+" ", str(getclassname(
    #         classIndex), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA))
    #     cv2.putText(imgOriginal, str(probability) + " ",  (180, 75),
    #                 font, 0.75, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow("Result", imgOriginal)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# キャプチャをリリースして、ウィンドウをすべて閉じる
cap.release()
cv2.destroyAllWindows()
