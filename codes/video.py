# import h5py
import pickle
from statistics import mode
from tabnanny import check
import numpy as np
import cv2
import os
from pathlib import Path
import json
import pandas as pd
import keras
from PIL import Image
import tensorflow as tf

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

# read meta.csv for getting the label
meta_df = pd.read_csv(f"{parent_path}/image_datasets/Meta.csv")

# import json
json_f = open(f"{parent_path}/keys.json", "r")
print(json_f)
parameters = json.load(json_f)
BACKUP_DIR = parameters["backup_dir"]
backup_dir = f"{parent_path }/{BACKUP_DIR}"
IMAGE_SIZE = parameters["image_size"]


# set up the video cam
cap = cv2.VideoCapture(0)
cap.set(3, framewidth)
cap.set(4, frameheight)
cap.set(10, brightness)

# import trained model
h5_in = open(f"{backup_dir}/my_model.h5", "rb")
# model = h5py.File(h5_in)
model = keras.models.load_model(f"{backup_dir}/my_model.h5", compile=False)
check_point_path = f"{backup_dir}/weights.ckpt"
check_point_dir = os.path.dirname(check_point_path)
model.load_weights(tf.train.latest_checkpoint(check_point_dir))


def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):
    img = cv2.equalizeHist(img)
    return img


def preprocessing(img):
    # Convert the captured frame into RGB
    img = Image.fromarray(img, mode='RGB')

    # Resizing into dimensions you used while training
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(img)

    # Expand dimensions to match the 4D Tensor shape.
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def getclassname(classNo):
    classNo = np.asarray(classNo, dtype=str)
    classNo = classNo[0]
    for i in range(len(meta_df)):
        if str(classNo) == str(meta_df.at[i, "ClassId"]):
            got_label = str(meta_df.at[i, "label"])
            return str(meta_df.at[i, "label"])


while True:
    # read image data
    success, frame = cap.read()

    # PROCESS IMAGE
    # img = preprocessing(img)

    img = preprocessing(frame)

    #cv2.imshow("Processed Image", img)
    #img = img.reshape(1, int(IMAGE_SIZE), int(IMAGE_SIZE), 1)

    cv2.putText(frame, "Class: ", (20, 35), font,
                0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Probability: ", (20, 75),
                font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    # Predict Image
    predictions = model.predict(img)
    classIndex = np.argmax(predictions, axis=1)
    probability = np.amax(predictions)

    if probability > threshold:
        cv2.putText(frame, str(classIndex)+" "+str(getclassname(classIndex)),
                    (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, str(round(probability*100, 2)) + "%",  (180, 75),
                    font, 0.75, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow("Result", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# キャプチャをリリースして、ウィンドウをすべて閉じる
cap.release()
cv2.destroyAllWindows()
