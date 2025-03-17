import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from imblearn.over_sampling import RandomOverSampler




class FetchFiles:
    def extraction(self, dataset_path, sample_size=None):

        # Load metadata
        files = os.listdir(dataset_path)
        metadata_path = os.path.join(dataset_path, files[2])
        df = pd.read_csv(metadata_path)

        # Turn cols in CSV to array
        image_ids = df['image_id'].tolist()
        labels = df['dx'].tolist()

        # Dict comprehension to label each type of cancer
        label_mapping = {label: idx for idx, label in enumerate(set(labels))}
        numerical_labels = [label_mapping[label] for label in labels]

        # The file path locally for both folders containing images
        images_folder_part_1 = os.path.join(dataset_path, files[0])
        images_folder_part_2 = os.path.join(dataset_path, files[1])

        # Take these folders and add to list
        image_paths = []
        for img_id in image_ids:
            part1_path = os.path.join(images_folder_part_1, img_id + ".jpg")
            part2_path = os.path.join(images_folder_part_2, img_id + ".jpg")
            if os.path.exists(part1_path):
                image_paths.append(part1_path)
            elif os.path.exists(part2_path):
                image_paths.append(part2_path)

        # Turn images to arrays with labels
        image_paths = np.array(image_paths)
        numerical_labels = np.array(numerical_labels)

        # Stratified split
        # Following the 80/20 split rule
        train_images, test_images, train_labels, test_labels = train_test_split(
            image_paths,
            numerical_labels,
            test_size=0.2,
            stratify=numerical_labels,
            random_state=42
        )

        oversample = RandomOverSampler(random_state=42)
        image_paths_resampled, numerical_labels_resampled = oversample.fit_resample(
            image_paths.reshape(-1, 1), numerical_labels
        )
        image_paths_resampled = image_paths_resampled.flatten

        # Transform into dataframes using pandas
        train_df = pd.DataFrame({
            "filename": train_images,
            "class": train_labels.astype(str)
        })

        test_df = pd.DataFrame({
            "filename": test_images,
            "class": test_labels.astype(str)
        })

        IMG_SIZE = (128, 128)
        BATCH_SIZE = 32

        # Data normalisation + augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_dataframe(
            dataframe=train_df,
            x_col="filename",
            y_col="class",
            target_size=IMG_SIZE,
            class_mode="categorical",
            batch_size=BATCH_SIZE
        )

        test_generator = test_datagen.flow_from_dataframe(
            dataframe=test_df,
            x_col="filename",
            y_col="class",
            target_size=IMG_SIZE,
            class_mode="categorical",
            batch_size=BATCH_SIZE
        )


        return train_generator, test_generator, label_mapping
