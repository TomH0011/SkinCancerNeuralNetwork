from tensorflow.keras import layers, Model, Sequential
import matplotlib.pyplot as plt
from tensorflow.keras.layers import MaxPooling2D

class PlotMetrics:
    @staticmethod
    def plot_training_history(self, history):
        """Plot comprehensive training metrics."""
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        # Plot accuracy
        axs[0, 0].plot(history.history['accuracy'], label='Train Accuracy')
        axs[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axs[0, 0].set_title('Model Accuracy')
        axs[0, 0].set_ylabel('Accuracy')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].legend()

        # Plot loss
        axs[0, 1].plot(history.history['loss'], label='Train Loss')
        axs[0, 1].plot(history.history['val_loss'], label='Validation Loss')
        axs[0, 1].set_title('Model Loss')
        axs[0, 1].set_ylabel('Loss')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].legend()

        # Plot recall
        axs[1, 0].plot(history.history['recall'], label='Train Recall')
        axs[1, 0].plot(history.history['val_recall'], label='Validation Recall')
        axs[1, 0].set_title('Model Recall')
        axs[1, 0].set_ylabel('Recall')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].legend()

        # Plot precision
        axs[1, 1].plot(history.history['precision'], label='Train Precision')
        axs[1, 1].plot(history.history['val_precision'], label='Validation Precision')
        axs[1, 1].set_title('Model Precision')
        axs[1, 1].set_ylabel('Precision')
        axs[1, 1].set_xlabel('Epoch')
        axs[1, 1].legend()

        plt.tight_layout()
        plt.show()