import seaborn as sns
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import shutil
import math
import cv2
from IPython.display import Image as IPImage
from IPython.display import Image, display_png
from PIL import Image


def pick_train_sample(labels, train_data):
    print('## Train Data Sample test##')
    NUM_CLASSES = len(labels)
    print_flg = np.zeros(NUM_CLASSES, dtype=int)
    all_printed = np.ones(NUM_CLASSES, dtype=int)
    fig = plt.figure(figsize=(64, 64), tight_layout=True)
    ax = []
    for i in range(NUM_CLASSES):
        ax.append(fig.add_subplot(1, NUM_CLASSES, i + 1))
    print(len(train_data))
    print(train_data)
    for a in range(len(train_data)):
        (image_data, label_data) = next(train_data)
        for b in range(len(label_data)):
            # print('a:{} b:{}'.format(a, b))
            # print(print_flg)
            # print(label_data[b])
            im = Image.open(image_data)
            im.show()
            expect = label_data[b].argmax()
            if print_flg[expect] == 0:
                print(expect)
                ax[expect].set_title(labels[expect])
                ax[expect].imshow(image_data[b])
                ax[expect].axis('off')

                print_flg[expect] = 1
        if np.array_equal(print_flg, all_printed):
            break

    plt.show()


def vis_failed_pic(model, labels, validation_data, val_dir, BATCH_SIZE, MAX_DISP):
    pred_data = model.predict(validation_data)
    print('Prediction data')
    pred_round = np.array(pred_data)
    print(np.round(pred_round, 3))
    print(pred_data.shape)

    #MAX_DISP = 100
    disp = 0
    i = 0
    correct_files = []
    incorrect_files = []
    validation_data.reset()
    for a in range(math.ceil(pred_data.shape[0] / BATCH_SIZE)):
        image_data, label_data = next(validation_data)
        for b in range(len(label_data)):
            expect = label_data[b].argmax()
            pred = pred_data[i].argmax()
            print('#{} Expected:{} -> Predicted:{}'.format(i,
                  labels[expect], labels[pred]))
            print('File: ' + validation_data.filenames[i])
            print('Each Prediction Value: ', end='')
            print(np.round(pred_data[i], 3))
            if expect != pred:
                print('**** INCORRECT ****')
                incorrect_files.append(
                    val_dir + '/' + validation_data.filenames[i])
                plt.imshow(image_data[b])
                plt.axis('off')
                plt.show()
                disp += 1
                if disp == MAX_DISP:
                    break
            else:
                print('* Correct *')
                correct_files.append(
                    val_dir + '/' + validation_data.filenames[i])
            i += 1

    with open('correct.txt', 'w') as f:
        for i in correct_files:
            f.write("%s\n" % i)
    with open('incorrect.txt', 'w') as f:
        for i in incorrect_files:
            f.write("%s\n" % i)


def vis_loss_accuracy(history, EPOCHS):
    # Visualize loss
    print('Loss')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Training and validation loss')
    plt.ylabel('loss')
    plt.xlim([0.0, EPOCHS])
    plt.xlabel('epoch')
    plt.legend(['loss', 'val_loss'], loc='upper left')
    plt.show()

    # Visualize accuracy
    print('Accuracy')
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Training and validation accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.xlim([0.0, EPOCHS])
    plt.ylim([0.0, 1.0])
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.show()


def vis_confusion_matrix(model, validation_data):
    validation_data.reset()
    validation_data.shuffle = False
    validation_data.batch_size = 1

    # Retrieve the first batch from the validation data
    for validation_image_batch, validation_label_batch in validation_data:
        break

    #predicted = model.predict_generator(validation_data, steps=validation_data.n)
    predicted = model.predict(validation_data, steps=validation_data.n)
    predicted_classes = np.argmax(predicted, axis=-1)

    # Apply normalization
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    cm = confusion_matrix(validation_data.classes, predicted_classes)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(12, 9))

    # https://seaborn.pydata.org/generated/seaborn.heatmap.html
    # https://matplotlib.org/users/colormaps.html
    sns.heatmap(cm, annot=True, square=True, cmap=plt.cm.Blues,
                xticklabels=validation_data.class_indices,
                yticklabels=validation_data.class_indices)

    print("Confusion Matrix")
    plt.title("Confusion Matrix")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.xlim([0.0, len(validation_data.class_indices)])
    plt.ylim([0.0, len(validation_data.class_indices)])
    plt.show()


def vis_hidden_layer(file_name, layer_name, image_size=64):
    train_dir = 'target_datasets/train'
    val_dir = 'target_datasets/val'

    backup_dir = './model'

    # Load model
    save_model_path = os.path.join(backup_dir, 'my_model.h5')
    model = tf.keras.models.load_model(save_model_path)
    summary = model.summary()

    plot_model(model, show_shapes=True)

    for layer in model.layers:
        if layer.name == layer_name:
            hidden_channel_amount = layer.output_shape[len(
                layer.output_shape)-1]
            print("hidden_channel_amount = " + str(hidden_channel_amount))
    ROW = 8
    hidden_layer_model = Model(
        inputs=model.input, outputs=model.get_layer(layer_name).output)

    val_dir_list = []
    with open(file_name, 'r') as f:
        #val_dir_list = list(f)
        for line in f:
            val_dir_list.append(line.strip())

    # Plot hidden layer
    COL = 8
    ROW = math.ceil(hidden_channel_amount / COL)
    for i, val_dir_index in enumerate(val_dir_list):
        img = cv2.imread(val_dir_index)
        img = cv2.resize(img, (image_size, image_size))
        target = np.reshape(
            img, (1, img.shape[0], img.shape[1], img.shape[2])).astype('float') / 255.0

        hidden_output = hidden_layer_model.predict(target)

        plt.figure(figsize=(20, 20 * ROW / COL))
        for j in range(hidden_channel_amount):
            plt.subplot(ROW, COL, j+1)
            plt.xticks([])
            plt.yticks([])
            plt.xlabel(f'#{j}')
            plt.imshow(hidden_output[0, :, :, j], cmap='viridis')
        print(val_dir_index)
        plt.show()


def vis_filter(model, layer):
    print('Visualize filter: ' + layer)
    vi_layer = model.get_layer(layer)

    # Get weights
    target_layer = vi_layer.get_weights()[0]
    filter_num = target_layer.shape[3]

    # Plot filter value
    COL = 8
    plt.figure()
    for i in range(filter_num):
        plt.subplot(math.ceil(filter_num / COL), COL, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(f'#{i}')
        plt.imshow(target_layer[:, :, 0, i], cmap="gray")

    plt.show()
