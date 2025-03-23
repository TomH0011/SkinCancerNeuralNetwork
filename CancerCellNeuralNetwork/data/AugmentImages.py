from tensorflow.keras.preprocessing.image import ImageDataGenerator

class AugmentImages:
    def __init__(self):
        # Data normalisation + augmentation
        self.train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )
        self.test_datagen = ImageDataGenerator(rescale=1. / 255)

    def get_generators(self):
        return self.train_datagen, self.test_datagen
