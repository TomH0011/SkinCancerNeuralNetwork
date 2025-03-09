import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import keras

class Build:
    def extraction(self):
        # Stored locally
        dataset_path = r"C:\Users\tomjh\Desktop\Cancer Cell Dataset"

        files = os.listdir(dataset_path)

        metadata_path = os.path.join(dataset_path, files[2])
        df = pd.read_csv(metadata_path)

        image_ids = df['image_id'].tolist()
        labels = df['dx'].tolist()

        label_mapping = {label: idx for idx, label in enumerate(set(labels))}
        numerical_labels = [label_mapping[label] for label in labels]

        images_folder_part_1 = os.path.join(dataset_path, files[0])
        images_folder_part_2 = os.path.join(dataset_path, files[1])

        image_paths = []
        for img_id in image_ids:
            part1_path = os.path.join(images_folder_part_1, img_id + ".jpg")
            part2_path = os.path.join(images_folder_part_2, img_id + ".jpg")
            if os.path.exists(part1_path):
                image_paths.append(part1_path)
            elif os.path.exists(part2_path):
                image_paths.append(part2_path)

        image_paths = np.array(image_paths)
        numerical_labels = np.array(numerical_labels)

        split_idx = int(len(image_paths) * 0.8)
        train_images, test_images = image_paths[:split_idx], image_paths[split_idx:]
        train_labels, test_labels = numerical_labels[:split_idx], numerical_labels[split_idx:]

        train_df = pd.DataFrame({
            "filename": train_images,
            "class": train_labels.astype(str)  # Convert to string
        })

        test_df = pd.DataFrame({
            "filename": test_images,
            "class": test_labels.astype(str)  # Convert to string
        })

        IMG_SIZE = (128, 128)
        BATCH_SIZE = 32

        train_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_dataframe(
            dataframe=train_df,
            x_col="filename",
            y_col="class",
            target_size=IMG_SIZE,
            class_mode="categorical",
            subset="training"
        )

        test_generator = train_datagen.flow_from_dataframe(
            dataframe=train_df,
            x_col="filename",
            y_col="class",
            target_size=IMG_SIZE,
            class_mode="categorical",
            subset="validation"
        )

        return train_generator, test_generator, label_mapping

class CNN:
    def define_cnn(self, train_generator):
        num_classes = len(train_generator.class_indices)

        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.75),
            layers.Dense(num_classes, activation='softmax')
        ])

        model.summary()
        return model

    def learn(self, model, train_generator, test_generator, label_mapping):

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                      metrics=[
                          keras.metrics.FalsePositives(),
                          keras.metrics.Recall(),
                          keras.metrics.CategoricalAccuracy()
                      ])

        history = model.fit(train_generator, epochs=10, validation_data=test_generator)

        plt.plot(history.history['categorical_accuracy'], label='train_accuracy')
        plt.plot(history.history['val_categorical_accuracy'], label='val_accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

        return model.evaluate(test_generator, verbose=2)

if __name__ == "__main__":
    build_model = Build()
    train_generator, test_generator, label_mapping = build_model.extraction()
    cnn_model = CNN().define_cnn(train_generator)
    CNN().learn(cnn_model, train_generator, test_generator, label_mapping)