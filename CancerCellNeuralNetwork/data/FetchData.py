import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler


class FetchFiles:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def extraction(self):
        # Load metadata
        files = os.listdir(self.dataset_path)
        metadata_path = os.path.join(self.dataset_path, files[2])  # Ensure this is the metadata CSV
        df = pd.read_csv(metadata_path)

        # Map labels to numbers
        label_mapping = {label: idx for idx, label in enumerate(set(df['dx']))}
        numerical_labels = df['dx'].map(label_mapping).values

        # Locate image files
        images_folder_1 = os.path.join(self.dataset_path, files[0])
        images_folder_2 = os.path.join(self.dataset_path, files[1])

        image_paths = [
            os.path.join(images_folder_1, img + ".jpg") if os.path.exists(os.path.join(images_folder_1, img + ".jpg"))
            else os.path.join(images_folder_2, img + ".jpg")
            for img in df['image_id']
        ]

        # Train-test split (80/20)
        train_images, test_images, train_labels, test_labels = train_test_split(
            image_paths, numerical_labels, test_size=0.2, stratify=numerical_labels, random_state=42
        )

        # Oversampling
        oversample = RandomOverSampler(random_state=42)
        train_images_resampled, train_labels_resampled = oversample.fit_resample(
            np.array(train_images).reshape(-1, 1), train_labels
        )

        # Convert to DataFrame
        train_df = pd.DataFrame({"filename": train_images_resampled.flatten(), "class": train_labels_resampled.astype(str)})
        test_df = pd.DataFrame({"filename": test_images, "class": test_labels.astype(str)})

        return train_df, test_df, label_mapping
