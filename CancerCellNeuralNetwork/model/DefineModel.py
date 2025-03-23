import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

class DefineCNN:
    # This is the specifics of the model
    # 2 density layers
    # 2 pooling layers
    # 4 convolutional layers
    def __init__(self, input_shape=(128, 128, 3), num_classes=7):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_model(self):
        # Keepy density units some multiple of 2
        # 1 pooling layer for every 2 conv layer
        model = Sequential([
            Input(shape=self.input_shape),
            Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'),
            Conv2D(32, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
            Conv2D(64, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2), padding='same'),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(self.num_classes, activation='softmax')
        ])
        return model
