import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def preprocess(batch_size = 32):
    """
    Preprocesses the CIFAR-10 dataset.

    Returns:
        x_train: Preprocessed training data.
        y_train: Preprocessed training labels.
        x_test: Preprocessed test data.
        y_test: Preprocessed test labels.
    """
    # Load CIFAR-10 data from keras datasets and split into train and test sets
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)
    
    data_generator = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    train_generator = data_generator.flow(x_train, y_train, batch_size)
        
    return train_generator,x_train,y_train, x_test, y_test