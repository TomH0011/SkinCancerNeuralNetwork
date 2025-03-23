from tensorflow.keras import layers, Model, Sequential
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class PlotConfusionMatrix:
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