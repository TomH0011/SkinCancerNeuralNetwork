import tensorflow as tf


class SaveModel:
    # Just saves the machine locally, feel free to alter the name but also
    # Remember to change the model that's loaded in Main.py
    @staticmethod
    def save(model, filename="working_cnn_model.keras"):
        model.save(filename)
        print(f"Model saved as {filename}")
