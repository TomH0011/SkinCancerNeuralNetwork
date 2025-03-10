import DefineCNN
import FetchFiles

if __name__ == "__main__":
    build_model = FetchFiles.Build()
    train_generator, test_generator, label_mapping = build_model.extraction()

    dropout_rate = 0.8
    l1_reg = 0.001
    l2_reg = 0.01
    learning_rate = 0.0001
    batch_size = 64
    epochs = 100

    cnn_model = DefineCNN.CNN().define_cnn(train_generator, dropout_rate=dropout_rate, l1_reg=l1_reg, l2_reg=l2_reg)
    DefineCNN.CNN().learn(cnn_model, train_generator, test_generator, label_mapping, learning_rate=learning_rate,
                          batch_size=batch_size, epochs=epochs)
