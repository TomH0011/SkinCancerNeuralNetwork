from tensorflow.keras.preprocessing.image import ImageDataGenerator

class CreateGenerators:
    def __init__(self, img_size=(128, 128), batch_size=32):
        self.IMG_SIZE = img_size
        self.BATCH_SIZE = batch_size

    def traintestgenerators(self, train_datagen, train_df, test_datagen, test_df):
        train_generator = train_datagen.flow_from_dataframe(
            dataframe=train_df,
            x_col="filename",
            y_col="class",
            target_size=self.IMG_SIZE,
            class_mode="categorical",
            batch_size=self.BATCH_SIZE
        )

        test_generator = test_datagen.flow_from_dataframe(
            dataframe=test_df,
            x_col="filename",
            y_col="class",
            target_size=self.IMG_SIZE,
            class_mode="categorical",
            batch_size=self.BATCH_SIZE
        )

        return train_generator, test_generator
