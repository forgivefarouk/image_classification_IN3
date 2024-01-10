import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from data_preprocessing import preprocess
from tensorflow.keras.optimizers import SGD

class CNNModel:
    def __init__(self):
        """
        Initializes the CNNModel object.
        """
        self.model = None
        self.history = None
        
    def build_model(self):
        """
        Build and compile a convolutional neural network model.

        Returns:
            model (tensorflow.keras.models.Sequential): The compiled model.
        """
        # Set the input shape and kernel size
        INPUT_SHAPE = (32, 32, 3)
        KERNEL_SIZE = (3, 3)

        # Create the model
        model = Sequential()

        # Add convolutional layers
        model.add(Conv2D(filters=32, kernel_size=KERNEL_SIZE, input_shape=INPUT_SHAPE, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=32, kernel_size=KERNEL_SIZE, input_shape=INPUT_SHAPE, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(filters=64, kernel_size=KERNEL_SIZE, input_shape=INPUT_SHAPE, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=64, kernel_size=KERNEL_SIZE, input_shape=INPUT_SHAPE, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(filters=128, kernel_size=KERNEL_SIZE, input_shape=INPUT_SHAPE, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=128, kernel_size=KERNEL_SIZE, input_shape=INPUT_SHAPE, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # Flatten the output from the convolutional layers
        model.add(Flatten())

        # Add dense layers
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.25)) 
        model.add(Dense(10, activation='softmax'))

        # Define the metrics
        METRICS = ['accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')]

        # Initialize the optimizer
        opt = SGD(learning_rate=0.01, momentum=0.9)
    
        # Compile the model
        model.compile(optimizer=opt,
                    loss='categorical_crossentropy',
                    metrics=METRICS)

        self.model = model
        return model

    def fit_model(self, train_generator,x_train, x_test, y_test, epochs=10, batch_size=32):
        """
        Fits the model to the training data.

        Args:
            x_train (ndarray): The training input data.
            y_train (ndarray): The training target data.
            x_test (ndarray): The testing input data.
            y_test (ndarray): The testing target data.
            epochs (int, optional): The number of epochs to train for. Defaults to 10.
            batch_size (int, optional): The batch size for training. Defaults to 64.

        Returns:
            history (tensorflow.keras.callbacks.History): The training history.
        """
        # Train the model
        history = self.model.fit(
            train_generator, 
            epochs=epochs, 
            steps_per_epoch = x_train.shape[0] // batch_size,
            validation_data=(x_test, y_test)
        )

        self.history = history
        return history

    def visualize_accuracy_loss(self):
        """
        Visualizes the accuracy and loss of the model.

        Returns:
            None
        """
        # Check if the model has been trained
        if self.history is None:
            print("Model has not been trained. Call fit_model first.")
            return

        # Visualize the accuracy
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        # Create a new figure for loss visualization
        plt.figure()
        
        # Visualize the loss
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

        # Show the plots
        plt.show()

if __name__ == "__main__":
    from tensorflow.keras.utils import plot_model
    batch_size = 32
    train_generator,x_train,y_train, x_test, y_test = preprocess(batch_size)
    model = CNNModel()
    model.build_model()
    model.fit_model(train_generator,x_train, x_test, y_test, epochs=1, batch_size=batch_size)
