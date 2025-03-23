import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

class RunModel:
    def __init__(self, learning_rate=0.0001, batch_size=64, epochs=50):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

    def train(self, model, train_generator, test_generator, label_mapping, class_weights=None):
        # Define callbacks
        # Early stopping to try to avoid model getting worse
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
        ]

        # Compile model
        # Recall and precision for false negative tracking
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=[
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.FalseNegatives(name='false_negatives'),
                tf.keras.metrics.CategoricalAccuracy(name='accuracy')
            ]
        )

        # Train
        # fitting the model :)
        history = model.fit(
            train_generator,
            epochs=self.epochs,
            validation_data=test_generator,
            callbacks=callbacks,
            class_weight=class_weights
        )

        return history
