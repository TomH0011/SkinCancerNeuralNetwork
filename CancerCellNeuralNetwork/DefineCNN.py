import tensorflow as tf
from keras import Input
from keras.src.layers import Conv2D, Flatten, Dense
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow.keras.layers import MaxPooling2D


class CNN:
    def define_cnn(self, train_generator, dropout_rate=0.5, l2_reg=0.001):

        # Create a more robust model architecture with proper regularization
        model = Sequential([
            Input(shape=(128, 128, 3)),  # Input layer defining the shape
            Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'),
            Conv2D(32, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),  # Corrected name
            Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
            Conv2D(64, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2), padding='same'),  # Corrected name
            Flatten(),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(7, activation='softmax')
        ])

        model.summary()
        return model

    def learn(self, model, train_generator, test_generator, label_mapping,
              learning_rate=0.0001, batch_size=32, epochs=50, class_weights=None):

        # Define callbacks for better training
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]

        # Use a more appropriate optimizer with a reasonable learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Compile with metrics focusing on medical relevance
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            metrics=[
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.FalseNegatives(name='false_negatives'),
                tf.keras.metrics.CategoricalAccuracy(name='accuracy')
            ]
        )

        # Train with class weights if provided
        history = model.fit(
            train_generator,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=test_generator,
            callbacks=callbacks,
            class_weight=class_weights
        )

        # Plot training metrics
        self.plot_training_history(history)

        # Evaluate and display detailed metrics
        return self.evaluate_model(model, test_generator, label_mapping)

    def plot_training_history(self, history):
        """Plot comprehensive training metrics."""
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        # Plot accuracy
        axs[0, 0].plot(history.history['accuracy'], label='Train Accuracy')
        axs[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axs[0, 0].set_title('Model Accuracy')
        axs[0, 0].set_ylabel('Accuracy')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].legend()

        # Plot loss
        axs[0, 1].plot(history.history['loss'], label='Train Loss')
        axs[0, 1].plot(history.history['val_loss'], label='Validation Loss')
        axs[0, 1].set_title('Model Loss')
        axs[0, 1].set_ylabel('Loss')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].legend()

        # Plot recall
        axs[1, 0].plot(history.history['recall'], label='Train Recall')
        axs[1, 0].plot(history.history['val_recall'], label='Validation Recall')
        axs[1, 0].set_title('Model Recall')
        axs[1, 0].set_ylabel('Recall')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].legend()

        # Plot precision
        axs[1, 1].plot(history.history['precision'], label='Train Precision')
        axs[1, 1].plot(history.history['val_precision'], label='Validation Precision')
        axs[1, 1].set_title('Model Precision')
        axs[1, 1].set_ylabel('Precision')
        axs[1, 1].set_xlabel('Epoch')
        axs[1, 1].legend()

        plt.tight_layout()
        plt.show()

    def evaluate_model(self, model, test_generator, label_mapping):
        """Evaluate model with detailed metrics and visualisations."""
        # Reset the generator to start from beginning
        test_generator.reset()

        # Get predictions
        y_pred_probs = model.predict(test_generator)
        y_pred = np.argmax(y_pred_probs, axis=1)

        # Get true labels
        y_true = test_generator.classes

        # Get class names
        class_indices = test_generator.class_indices
        class_names = {v: k for k, v in class_indices.items()}

        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=[class_names[i] for i in range(len(class_names))]))

        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[class_names[i] for i in range(len(class_names))],
                    yticklabels=[class_names[i] for i in range(len(class_names))])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

        # Standard evaluation metrics
        return model.evaluate(test_generator, verbose=1)