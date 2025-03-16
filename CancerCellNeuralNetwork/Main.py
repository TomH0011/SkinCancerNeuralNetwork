from CancerCellNeuralNetwork.DefineCNN import CNN
from CancerCellNeuralNetwork.FetchFiles import FetchFiles
from FileUpload import FileUpload


def test_model():
    file_uploader = FileUpload()
    prediction, certainty = file_uploader.test_one_image(uploaded_file_path=r"C:\Users\tomjh\Desktop")
    file_uploader.final_score(prediction, certainty)

# Only run this if needed
if __name__ == "__main__":
    train_model = False  # Change this to False if you just want to test an image

    if train_model:
        dataset_path = r"C:\Users\tomjh\Desktop\Cancer Cell Dataset"
        data_prep = FetchFiles()
        train_generator, test_generator, label_mapping = data_prep.extraction(
            dataset_path=dataset_path,
            sample_size=None
        )

        dropout_rate = 0.55
        learning_rate = 0.0001
        batch_size = 64
        epochs = 50

        cnn = CNN()
        model = cnn.define_cnn(train_generator, dropout_rate=dropout_rate)
        model.save('working_cnn_model.keras')
        cnn.learn(
            model,
            train_generator,
            test_generator,
            label_mapping,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
        )

    test_model()  # Run just this when needed
