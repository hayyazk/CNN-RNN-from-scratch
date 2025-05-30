import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from sklearn.metrics import f1_score
from conv_layer import Conv2D
from max_pooling import MaxPooling2D
from activation import relu, softmax
from dense_layer import Dense, Flatten

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_train = y_train.flatten()
y_test = y_test.flatten()

# testing subset data (1000 data latih dan 10 data uji)
x_train, y_train = x_train[:1000], y_train[:1000]
x_test, y_test = x_test[:10], y_test[:10]

keras_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(8, 3, activation='relu', padding='same', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

keras_model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

keras_model.fit(x_train, y_train, epochs=3, batch_size=32, verbose=1)

keras_preds = keras_model.predict(x_test).argmax(axis=1)
f1_keras = f1_score(y_test, keras_preds, average='macro')
print("Macro F1-score (Keras):", f1_keras)

# ekstraksi bobot dari layer-layer
w1, b1 = keras_model.layers[0].get_weights()  # Conv2D(8)
w2, b2 = keras_model.layers[2].get_weights()  # Conv2D(16)
w3, b3 = keras_model.layers[5].get_weights()  # Dense(32)
w4, b4 = keras_model.layers[6].get_weights()  # Dense(10)

x = x_test
x = Conv2D(w1, b1).forward(x)
x = relu(x)
x = MaxPooling2D().forward(x)

x = Conv2D(w2, b2).forward(x)
x = relu(x)
x = MaxPooling2D().forward(x)

x = Flatten().forward(x)
x = Dense(w3, b3).forward(x)
x = relu(x)
x = Dense(w4, b4).forward(x)
predictions = softmax(x)

predicted_labels = predictions.argmax(axis=1)
f1_manual = f1_score(y_test, predicted_labels, average='macro')
print("Macro F1-score (manual forward):", f1_manual)

