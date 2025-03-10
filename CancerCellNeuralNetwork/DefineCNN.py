import tensorflow as tf
from keras import Model
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import keras
from tensorflow.keras.applications import ResNet50

class CNN:
    def define_cnn(self, train_generator, dropout_rate=0.5, l1_reg=0.001, l2_reg=0.01):
        # Transfer learning with ResNet50
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation='relu')(x)
        predictions = layers.Dense(len(train_generator.class_indices), activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        # Freeze the base model layers
        for layer in base_model.layers:
            layer.trainable = False

        model.summary()
        return model

    def learn(self, model, train_generator, test_generator, label_mapping, learning_rate=0.0001, batch_size=32,
              epochs=100):
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                      metrics=[
                          keras.metrics.FalseNegatives(),
                          keras.metrics.Recall(),
                          keras.metrics.CategoricalAccuracy()
                      ])

        history = model.fit(train_generator,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=test_generator)

        plt.plot(history.history['categorical_accuracy'], label='train_accuracy')
        plt.plot(history.history['val_categorical_accuracy'], label='val_accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

        return model.evaluate(test_generator, verbose=2)