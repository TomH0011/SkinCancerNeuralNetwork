import os
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model

class UploadTestImage:
    def test_one_image(self, uploaded_file_path):

        IMG_SIZE = (128, 128)

        os.makedirs(uploaded_file_path, exist_ok=True)  # Ensure directory exists
        img_path = os.path.join(uploaded_file_path, "Melanoma_skin_lesion.jpg")

        img = image.load_img(img_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        model = load_model(r"D:\  Hello World\CancerCellNeuralNetwork\CancerCellNeuralNetwork\working_cnn_model.keras")
        predictions = model.predict(img_array)
        predicted_classification = np.argmax(predictions)
        certainty = np.max(predictions)

        return predicted_classification, certainty

    def final_score(self, predicted_classification, certainty):
        print(f'The prediction is {predicted_classification} with {certainty} certainty')
        print('Mapping:')
        print('0 → akiec (Actinic keratosis)')
        print('1 → bcc (Basal cell carcinoma)')
        print('2 → bkl (Benign keratosis-like lesions)')
        print('3 → df (Dermatofibroma)')
        print('4 → mel (Melanoma)')
        print('5 → nv (Nevi)')
        print('6 → vasc (Vascular lesions)')