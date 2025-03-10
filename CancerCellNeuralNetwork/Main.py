from FetchFiles import FetchFiles
from DefineCNN import CNN

if __name__ == "__main__":
    # Use more data for training
    dataset_path = r"C:\Users\tomjh\Desktop\Cancer Cell Dataset"
    data_prep = FetchFiles()
    train_generator, test_generator, label_mapping = data_prep.extraction(
        dataset_path=dataset_path,
        sample_size=2000  # Use more data, or None for all available data
    )

    # Define model with better hyperparameters
    dropout_rate = 0.55
    learning_rate = 0.0001
    batch_size = 64
    epochs = 50

    # Create and train model
    cnn = CNN()
    model = cnn.define_cnn(train_generator, dropout_rate=dropout_rate)
    cnn.learn(
        model, 
        train_generator, 
        test_generator, 
        label_mapping, 
        learning_rate=learning_rate,
        batch_size=batch_size, 
        epochs=epochs,
    )