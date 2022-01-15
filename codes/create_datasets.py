import os
from pathlib import Path
import shutil
import json
import sys
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt

cwd_path = os.path.dirname(__file__)
parent_path = Path(cwd_path).parent

json_f = open(f"{parent_path}/keys.json", "r")
parameters = json.load(json_f)
TRAIN_DIR = f'{parent_path}/{parameters["train_dir"]}'
TEST_DIR = f'{parent_path}/{parameters["test_dir"]}'

print(TEST_DIR)
print(TRAIN_DIR)


test_csv = pd.read_csv(f"{parent_path}/image_datasets/Test.csv")
train_csv = pd.read_csv(f"{parent_path}/image_datasets/Train.csv")


def main(test_num, train_num, class_list):
    if os.path.exists(TRAIN_DIR):
        shutil.rmtree(TRAIN_DIR)
    os.makedirs(TRAIN_DIR)

    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.makedirs(TEST_DIR)

    class_list = class_list.split(",")
    print(class_list)

    # test_data
    print("########creating test data##############")
    possible_img = []
    for row in range(len(test_csv)):
        if str(test_csv["ClassId"][row]) in class_list:
            possible_img.append(
                [test_csv["Path"][row], test_csv["ClassId"][row]])

    ns = []
    test_figure = plt.figure()
    class_count = np.array([0 for _ in range(len(class_list))])
    bar_x = np.array(class_list)

    for class_num in class_list:
        os.makedirs(f"{TEST_DIR}/{class_num}")

    if len(possible_img) > int(test_num):
        while len(ns) < int(test_num):
            n = random.randint(0, len(possible_img)-1)
            if not n in ns:
                ns.append(n)
                img_file_name = possible_img[n][0].replace("Test/", "")
                class_id = possible_img[n][1]
                class_count[int(possible_img[n][1]) - 1] += 1
                # print(img_file_name)
                shutil.copy(
                    f"{parent_path}/image_datasets/{possible_img[n][0]}", f"{TEST_DIR}/{class_id}/{img_file_name}")
    else:
        while len(ns) < len(possible_img):
            n = random.randint(0, len(possible_img)-1)
            if not n in ns:
                ns.append(n)
                img_file_name = possible_img[n][0].replace("Test/", "")
                class_id = possible_img[n][1]
                class_count[int(possible_img[n][1]) - 1] += 1
                # print(img_file_name)
                shutil.copy(
                    f"{parent_path}/image_datasets/{possible_img[n][0]}", f"{TEST_DIR}/{class_id}/{img_file_name}")
    plt.bar(bar_x, class_count)
    plt.xlabel("class name")
    plt.title("test data count")
    # plt.show()
    test_figure.savefig(f"{parent_path}/image_datasets/Test_Used_Count.png")
    print("########test data created##############")

    print("########creating training data ##############")
    # train_data
    train_figure = plt.figure()
    class_count = np.array([0 for _ in range(len(class_list))])
    bar_x = np.array(class_list)
    for class_num in class_list:
        possible_img = []
        for row in range(len(train_csv)):
            if str(train_csv["ClassId"][row]) == class_num:
                possible_img.append(
                    [train_csv["Path"][row], train_csv["ClassId"][row]])

        os.makedirs(f"{TRAIN_DIR}/{class_num}")

        ns = []
        if len(possible_img) > int(train_num):
            while len(ns) < int(train_num):
                n = random.randint(0, len(possible_img)-1)
                if not n in ns:
                    ns.append(n)
                    img_file_name = possible_img[n][0].replace("Train/", "")
                    class_count[int(possible_img[n][1]) - 1] += 1
                    # print(img_file_name)
                    shutil.copy(
                        f"{parent_path}/image_datasets/{possible_img[n][0]}", f"{TRAIN_DIR}/{img_file_name}")
        else:
            while len(ns) < len(possible_img):
                n = random.randint(0, len(possible_img)-1)
                if not n in ns:
                    ns.append(n)
                    img_file_name = possible_img[n][0].replace("Train/", "")
                    class_count[int(possible_img[n][1]) - 1] += 1
                    # print(img_file_name)
                    shutil.copy(
                        f"{parent_path}/image_datasets/{possible_img[n][0]}", f"{TRAIN_DIR}/{img_file_name}")
    plt.bar(bar_x, class_count)
    plt.title("train data count")
    plt.xlabel("class name")
    # plt.show()
    train_figure.savefig(f"{parent_path}/image_datasets/Train_Used_Count.png")
    print("########training data created##############")


if __name__ == "__main__":
    param = sys.argv
    main(param[1], param[2], param[3])
