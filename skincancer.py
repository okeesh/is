import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Activation, MaxPooling2D
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from time import time

X = pickle.load(open("X.pickle", "rb"))
Y_cat = pickle.load(open("y.pickle", "rb"))

activations = ['sigmoid', 'tanh', 'leaky relu', 'relu']
optimizers = ['SGD', 'Adagrad', 'Adadelta', 'RMSprop', 'Adam']

batch_size = 16
epochs = 50

x_train, x_test, y_train, y_test = train_test_split(X, Y_cat, test_size=0.25, random_state=42)
for activation in activations:
    for optimizer in optimizers:
            NAME = "O-{}_A-{}".format(optimizer, activation)

            model = Sequential()

            model.add(Conv2D(256, (3, 3), activation=activation, input_shape=(32, 32, 3)))
            model.add(MaxPool2D(pool_size=(2, 2)))
            model.add(Dropout(0.3))

            model.add(Flatten())

            model.add(Dense(7, activation='softmax'))
            model.summary()
            tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
            model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

            model.fit(
                x_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test, y_test),
                verbose=2,
                callbacks=[tensorboard])
            score = model.evaluate(x_test, y_test)
            print('Test accuracy:', score[1])
            # Prediction on test data
            y_pred = model.predict(x_test)
            # Convert predictions classes to one hot vectors
            y_pred_classes = np.argmax(y_pred, axis=1)
            # Convert test data to one hot vectors
            y_true = np.argmax(y_test, axis=1)
            cm = confusion_matrix(y_true, y_pred_classes)


            fig, ax = plt.subplots(figsize=(10, 10))
            sns.set(font_scale=1.6)
            sns.heatmap(cm, annot=True, linewidths=2, ax=ax)
            plt.show()

            # PLot fractional incorrect misclassifications
            incorr_fraction = 1 - np.diag(cm) / np.sum(cm, axis=1)
            plt.bar(np.arange(7), incorr_fraction)
            plt.xlabel('True Label')
            plt.ylabel('Fraction of incorrect predictions')
            plt.show()
            print(NAME)

# # conv->pool->conv->pool
# # conv nimmt mehrere pixel und fasst sie in einem zusammen
# # Das Fenster shiftet zu den nächsten Pixel und macht weiter
# # pooling (maxPooling) findet das max value von den conv fenstern
# model = Sequential()
# model.add(Conv2D(256, (3, 3), activation="relu", input_shape=(32, 32, 3)))
# # model.add(BatchNormalization())
# model.add(MaxPool2D(pool_size=(2, 2)))
# model.add(Dropout(0.3))
#
# model.add(Conv2D(128, (3, 3), activation='relu'))
# # model.add(BatchNormalization())
# model.add(MaxPool2D(pool_size=(2, 2)))
# model.add(Dropout(0.3))
#
# model.add(Conv2D(64, (3, 3), activation='relu'))
# # model.add(BatchNormalization())
# model.add(MaxPool2D(pool_size=(2, 2)))
# model.add(Dropout(0.3))
# model.add(Flatten())
#
# model.add(Dense(32))
# model.add(Dense(7, activation='softmax'))
#
#
# # Train
# # You can also use generator to use augmentation during training.
# # softmax sagt aus zu wv. prozent das model denkt, dass es die klasse ist
#
#
#
# score = model.evaluate(x_test, y_test)
# print('Test accuracy:', score[1])
#
# # plot the training and validation accuracy and loss at each epoch
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(loss) + 1)
# plt.plot(epochs, loss, 'y', label='Training loss')
# plt.plot(epochs, val_loss, 'r', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
#
# acc = history.history['acc']
# val_acc = history.history['val_acc']
# plt.plot(epochs, acc, 'y', label='Training acc')
# plt.plot(epochs, val_acc, 'r', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()
#
# # Prediction on test data
# y_pred = model.predict(x_test)
# # Convert predictions classes to one hot vectors
# y_pred_classes = np.argmax(y_pred, axis=1)
# # Convert test data to one hot vectors
# y_true = np.argmax(y_test, axis=1)
#
# # Print confusion matrix
# cm = confusion_matrix(y_true, y_pred_classes)
#
#
# fig, ax = plt.subplots(figsize=(10, 10))
# sns.set(font_scale=1.6)
# sns.heatmap(cm, annot=True, linewidths=2, ax=ax)
#
# # PLot fractional incorrect misclassifications
# incorr_fraction = 1 - np.diag(cm) / np.sum(cm, axis=1)
# plt.bar(np.arange(7), incorr_fraction)
# plt.xlabel('True Label')
# plt.ylabel('Fraction of incorrect predictions')
# plt.show()
