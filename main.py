import zipfile
import kagglehub
import numpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import glob
from PIL import Image
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import Recall


# MY SECRET KEY "username":"tomhoward0","key":"a104396db2956a6f55db671e74ffc5c7"}
class build:
    def extraction(self):
        dataset_path = r"C:\Users\tomjh\Desktop\Cancer Cell Dataset"

        files = os.listdir(dataset_path)

        # Get metadata file (assuming it's in the 3rd position, files[2])
        metadata_path = os.path.join(dataset_path, files[2])
        df = pd.read_csv(metadata_path)

        # Extract image filenames and corresponding labels from CSV
        image_ids = df['image_id'].tolist()
        labels = df['dx'].tolist()  # 'dx' contains classification labels

        # Convert labels into numerical values (class indices)
        label_mapping = {label: idx for idx, label in enumerate(set(labels))}
        labels = [label_mapping[label] for label in labels]  # Convert each label to an index

        # Define directories for images (adjust if structure is different)
        images_folder_part_1 = os.path.join(dataset_path, files[0])
        images_folder_part_2 = os.path.join(dataset_path, files[1])

        # Create new lists for full image paths
        image_paths = []
        for img_id in image_ids:
            part1_path = os.path.join(images_folder_part_1, img_id + ".jpg")
            part2_path = os.path.join(images_folder_part_2, img_id + ".jpg")
            if os.path.exists(part1_path):
                image_paths.append(part1_path)
            elif os.path.exists(part2_path):
                image_paths.append(part2_path)

        # Convert lists to NumPy arrays
        image_paths = np.array(image_paths)
        labels = np.array(labels)

        # Split into training (80%) and testing (20%) sets
        split_idx = int(len(image_paths) * 0.8)
        train_images, test_images = image_paths[:split_idx], image_paths[split_idx:]
        train_labels, test_labels = labels[:split_idx], labels[split_idx:]

        # Define image size and batch size
        IMG_SIZE = (128, 128)  # Resize images to 128x128
        BATCH_SIZE = 32

        # Use ImageDataGenerator for batch loading
        train_datagen = ImageDataGenerator(rescale=1. / 255)  # Normalize pixel values
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        # Create generators for batch loading
        train_generator = train_datagen.flow_from_dataframe(
            dataframe=pd.DataFrame({"filename": train_images, "class": train_labels}),
            x_col="filename",
            y_col="class",
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="raw",  # Directly return numerical class labels
            shuffle=True
        )

        test_generator = test_datagen.flow_from_dataframe(
            dataframe=pd.DataFrame({"filename": test_images, "class": test_labels}),
            x_col="filename",
            y_col="class",
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="raw",
            shuffle=False
        )

        return train_generator, test_generator, label_mapping

    def draw(self, matrix_img):
        plt.figure(figsize=(10, 10))
        plt.imshow(matrix_img)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        # The CIFAR labels happen to be arrays,
        # which is why you need the extra index
        plt.show()


class cnn:
    def define_cnn(self):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        # To avoid overfitting
        model.add(layers.Dropout(0.75))
        model.add(layers.Dense(7, activation='softmax'))

        model.summary()
        return model

    def learn(self, model, train_generator, test_generator):
        print(f'the length of label mapping is: {len(label_mapping)}')

        print("Training data shape:",
              [x.shape for x in next(iter(train_generator))])
        print("Test data shape:",
              [x.shape for x in next(iter(test_generator))])

        # adam for default dataset as it's not too big, recall as false negatives are costly here
        # Current bug getting Recall to work
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['accuracy'])

        history = model.fit(train_generator, epochs=10, validation_data=test_generator)

        plt.plot(history.history['accuracy'], label='train_accuracy')
        plt.plot(history.history['val_accuracy'], label='val_Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

        return model.evaluate(test_generator, verbose=2)


if __name__ == "__main__":
    model = build()  # Create an instance of the class

    train_generator, test_generator, label_mapping = model.extraction()  # Get image matrix and dataframe
    model = cnn().define_cnn()
    cnn().learn(model, train_generator, test_generator)
