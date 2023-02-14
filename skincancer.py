import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from keras.callbacks import TensorBoard
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.models import Sequential
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

X = pickle.load(open("X.pickle", "rb"))
Y_cat = pickle.load(open("y.pickle", "rb"))

activations = ['relu']
optimizers = ['Adam']

batch_size = 16
epochs = 250


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

        history = model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, y_test),
            verbose=2,
            callbacks=[tensorboard])
        score = model.evaluate(x_test, y_test)
        print('Test accuracy:', score[1])
        y_pred = model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1)
        cm = confusion_matrix(y_true, y_pred_classes)

        fig, ax = plt.subplots(figsize=(10, 10))
        sns.set(font_scale=1.6)
        sns.heatmap(cm, annot=True, linewidths=2, ax=ax)
        plt.show()

        incorr_fraction = 1 - np.diag(cm) / np.sum(cm, axis=1)
        plt.bar(np.arange(7), incorr_fraction)
        plt.xlabel('True Label')
        plt.ylabel('Fraction of incorrect predictions')
        plt.show()

        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, 'y', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        acc = history.history['acc']
        val_acc = history.history['val_acc']
        plt.plot(epochs, acc, 'y', label='Training acc')
        plt.plot(epochs, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()
        print(NAME)

        test_loss, test_acc = model.evaluate(x_test, y_test)
        print('Test loss:', test_loss)
        print('Test accuracy:', test_acc)
        y_pred_labels = np.argmax(y_pred, axis=1)
        class_report = classification_report(np.argmax(y_test, axis=1), y_pred_labels, labels=[0, 1, 2, 3, 4, 5, 6])
        print(class_report)


# model = Sequential()
# model.add(Conv2D(256, (3, 3), activation="relu", input_shape=(32, 32, 3)))
# model.add(MaxPool2D(pool_size=(2, 2)))
# model.add(Dropout(0.3))
#
# model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(MaxPool2D(pool_size=(2, 2)))
# model.add(Dropout(0.3))
#
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPool2D(pool_size=(2, 2)))
# model.add(Dropout(0.3))
# model.add(Flatten())
#
# model.add(Dense(32))
# model.add(Dense(7, activation='softmax'))
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

# y_pred = model.predict(x_test)
# y_pred_classes = np.argmax(y_pred, axis=1)

# y_true = np.argmax(y_test, axis=1)

# cm = confusion_matrix(y_true, y_pred_classes)
#
#
# fig, ax = plt.subplots(figsize=(10, 10))
# sns.set(font_scale=1.6)
# sns.heatmap(cm, annot=True, linewidths=2, ax=ax)
#
# incorr_fraction = 1 - np.diag(cm) / np.sum(cm, axis=1)
# plt.bar(np.arange(7), incorr_fraction)
# plt.xlabel('True Label')
# plt.ylabel('Fraction of incorrect predictions')
# plt.show()
