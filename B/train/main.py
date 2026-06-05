from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np

X_train = np.loadtxt('../dataset/input.csv', delimiter=',')
Y_train = np.loadtxt('../dataset/labels.csv', delimiter=',')

X_test = np.loadtxt('../dataset/input_test.csv', delimiter=',')
Y_test = np.loadtxt('../dataset/labels_test.csv', delimiter=',')

X_train = X_train.reshape(len(X_train), 100, 100, 3)
Y_train = Y_train.reshape(len(Y_train), 1)

X_test = X_test.reshape(len(X_test), 100, 100, 3)
Y_test = Y_test.reshape(len(Y_test), 1)

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D((2,2)),

    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2,2)),

    Flatten(),

    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(X_train, Y_train, epochs=10, batch_size=32)