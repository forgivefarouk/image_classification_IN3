from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from data_preprocessing import preprocess
model_path = "models/cnn_model_100.h5"


def evaluate_model(model_path,x_test,y_test):
    """
    Evaluates the model on the test data.

    Args:
        model_path (str): The path to the model file.
        x_test (npy): The test input data.
        y_test (npy): The test target data.

    Returns:
        Tuple[float, float, float, float]: The test loss, test accuracy, precision, and recall.
    """
    # Load the model
    model = load_model(model_path)

    # Evaluate the model
    test_loss, test_acc, precision, recall = model.evaluate(x_test, y_test, verbose=0)

    return test_loss, test_acc, precision, recall
    


    
if __name__ == "__main__":
    import numpy as np
    _, _,_,x_test, y_test = preprocess()
    _, test_acc,_ , _ = evaluate_model(model_path, x_test, y_test)
    print("Test accuracy: {:.2f}%".format(test_acc * 100))

    
    
    