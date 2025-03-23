import tensorflow as tf

from CancerCellNeuralNetwork.data.FetchData import FetchFiles

DATASET_PATH = r"C:\Users\tomjh\Desktop\Cancer Cell Dataset"


class RunTestOnly:
    def run_test_only(self):
        # Load the pre-trained model
        model = tf.keras.models.load_model('working_cnn_model.keras')

        # You'll need to load or recreate the label_mapping here
        # This could be loaded from a saved file or recreated
        # For example:
        fetcher = FetchFiles(DATASET_PATH)
        _, _, label_mapping = fetcher.extraction()

        return model, label_mapping
