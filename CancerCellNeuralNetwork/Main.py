from data.FetchData import FetchFiles
from data.AugmentImages import AugmentImages
from data.CreateGenerators import CreateGenerators
from model.DefineModel import DefineCNN
from model.RunModel import RunModel
from model.SaveModel import SaveModel
from plotting.ConfusionMatrix import PlotConfusionMatrix
from plotting.ModelEvaluation import PlotMetrics
from testing.UploadTestImage import UploadTestImage
from testing.RunTestOnly import RunTestOnly
import tensorflow as tf

DATASET_PATH = r"C:\Users\tomjh\Desktop\Cancer Cell Dataset"
TEST_IMAGE_PATH = r"C:\Users\tomjh\Desktop"


def run_full_pipeline():
    # 1. Fetch and preprocess data
    fetcher = FetchFiles(DATASET_PATH)
    train_df, test_df, label_mapping = fetcher.extraction()

    # 2. Data augmentation
    augmentor = AugmentImages()
    train_datagen, test_datagen = augmentor.get_generators()

    # 3. Create generators
    generator_creator = CreateGenerators()
    train_generator, test_generator = generator_creator.traintestgenerators(
        train_datagen, train_df, test_datagen, test_df
    )

    # 4. Define CNN model
    cnn = DefineCNN()
    model = cnn.build_model()
    model.summary()  # Print model architecture

    # 5. Train Model
    trainer = RunModel(learning_rate=0.0001, batch_size=64, epochs=50)
    history = trainer.train(model, train_generator, test_generator, label_mapping)

    # 6. Save model
    SaveModel.save(model)

    # 7. Draw plots
    confusion_matrix = PlotConfusionMatrix()
    evaluation_metric = confusion_matrix.evaluate_model(model, test_generator, label_mapping)
    draw_metrics = PlotMetrics()
    metric_sheet = draw_metrics.plot_training_history(history)

    return model, label_mapping


# def run_test_only():
#     # Load the pre-trained model
#     model = tf.keras.models.load_model('working_cnn_model.keras')
#
#     # You'll need to load or recreate the label_mapping here
#     # This could be loaded from a saved file or recreated
#     # For example:
#     fetcher = FetchFiles(DATASET_PATH)
#     _, _, label_mapping = fetcher.extraction()
#
#     return model, label_mapping


if __name__ == '__main__':
    Full_Test = False
    runner = RunTestOnly()

    if Full_Test:
        model, label_mapping = run_full_pipeline()
    else:
        model, label_mapping = runner.run_test_only()

    # 8. Test with new image (runs in both modes)
    file_uploader = UploadTestImage()
    prediction, certainty = file_uploader.test_one_image(
        # You may need to pass the model to this function
        # You may need to pass the label mapping
        uploaded_file_path=TEST_IMAGE_PATH
    )
    file_uploader.final_score(prediction, certainty)