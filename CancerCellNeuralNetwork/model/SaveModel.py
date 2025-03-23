import tensorflow as tf


class SaveModel:
    @staticmethod
    def save(model, filename="working_cnn_model.keras"):
        model.save(filename)
        print(f"Model saved as {filename}")
