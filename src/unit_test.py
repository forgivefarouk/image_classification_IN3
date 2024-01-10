import unittest
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from model_building import CNNModel
from data_preprocessing import preprocess
from model_evaluation import evaluate_model
import cv2
from tensorflow.keras.models import load_model
from predict import predict

# Test the shapes of the preprocessed data
class Testpreprocess(unittest.TestCase):
    def setUp(self):
        self.train_generator,self.x_train, self.y_train, self.x_test, self.y_test = preprocess()

    print("Test X_train shape..")
    def test_x_train_shape(self):
        self.assertEqual(self.x_train.shape, (50000, 32, 32, 3))
    
    print("Test y_train shape..")        
    def test_y_train_shape(self):
        self.assertEqual(self.y_train.shape, (50000, 10))
    print("Test X_test shape..")
    def test_x_test_shape(self):
        self.assertEqual(self.x_test.shape, (10000, 32, 32, 3))
    print("Test y_test shape..")
    def test_y_test_shape(self):
        self.assertEqual(self.y_test.shape, (10000, 10))


class TestBuildModel(unittest.TestCase):
    def setUp(self):
        cnn = CNNModel()
        self.model = cnn.build_model()

    print("Test model architecture..")
    def test_model_architecture(self):
        # Check the number of layers in the model
        self.assertEqual(len(self.model.layers), 23)
        
        # Check the type of each layer
        self.assertIsInstance(self.model.layers[0], Conv2D)
        self.assertIsInstance(self.model.layers[1], BatchNormalization)
        self.assertIsInstance(self.model.layers[2], Conv2D)
        self.assertIsInstance(self.model.layers[3], BatchNormalization)
        self.assertIsInstance(self.model.layers[4], MaxPool2D)
        self.assertIsInstance(self.model.layers[5], Dropout)
        self.assertIsInstance(self.model.layers[6], Conv2D)
        self.assertIsInstance(self.model.layers[7], BatchNormalization)
        self.assertIsInstance(self.model.layers[8], Conv2D)
        self.assertIsInstance(self.model.layers[9], BatchNormalization)
        self.assertIsInstance(self.model.layers[10], MaxPool2D)
        self.assertIsInstance(self.model.layers[11], Dropout)
        self.assertIsInstance(self.model.layers[12], Conv2D)
        self.assertIsInstance(self.model.layers[13], BatchNormalization)
        self.assertIsInstance(self.model.layers[14], Conv2D)
        self.assertIsInstance(self.model.layers[15], BatchNormalization)
        self.assertIsInstance(self.model.layers[16], MaxPool2D)
        self.assertIsInstance(self.model.layers[17], Dropout)
        self.assertIsInstance(self.model.layers[18], Flatten)
        self.assertIsInstance(self.model.layers[19], Dense)
        self.assertIsInstance(self.model.layers[20], BatchNormalization)
        self.assertIsInstance(self.model.layers[21], Dropout)
        self.assertIsInstance(self.model.layers[22], Dense)

    print("Test model input shape..")
    def test_model_input_shape(self):
        # Check the input shape of the first layer
        self.assertEqual(self.model.layers[0].input_shape, (None, 32, 32, 3))

    print("Test model output shape..")
    def test_model_output_shape(self):
        # Check the output shape of the last layer
        self.assertEqual(self.model.layers[-1].output_shape, (None, 10))




class TestEvaluateModel(unittest.TestCase):
    
    print("Test model evaluation..")
    def test_evaluate_model(self):
        _, _,_, x_test, y_test = preprocess()
        test_loss, test_acc,_,_= evaluate_model("models/cnn_model_100.h5", x_test, y_test)
        self.assertGreater(test_acc, 0.5)
        
        

class TestPredict(unittest.TestCase):
    def setUp(self):
        self.model_path = "models/cnn_model_100.h5"
        self.img_path = "dog.jpg"

    print("Test loaded model..")
    def test_predict_loaded_model(self):
        # Test if the model is loaded successfully
        model = load_model(self.model_path)
        self.assertIsNotNone(model)

    print("Test loaded image..")
    def test_predict_loaded_image(self):
        # Test if the image is loaded successfully
        img = cv2.imread(self.img_path)
        self.assertIsNotNone(img)

    print("Test image shape..")
    def test_predict_image_shape(self):
        # Test if the image is resized to 32x32 pixels
        img = cv2.imread(self.img_path)
        img_resized = cv2.resize(img, (32, 32))
        self.assertEqual(img_resized.shape, (32, 32, 3))

    print("Test reshaped image shape..")
    def test_predict_reshaped_image(self):
        # Test if the image is reshaped for further processing
        img = cv2.imread(self.img_path)
        img_resized = cv2.resize(img, (32, 32))
        img_reshaped = img_resized.reshape(1, 32, 32, 3)
        self.assertEqual(img_reshaped.shape, (1, 32, 32, 3))

    print("Test predicted label..")
    def test_predict_label_prediction(self):
        # Test if the predicted label is within the defined labels
        labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
          'dog', 'frog', 'horse', 'ship', 'truck']
        img = cv2.imread(self.img_path)
        img_resized = cv2.resize(img, (32, 32))
        img_reshaped = img_resized.reshape(1, 32, 32, 3)
        model = load_model(self.model_path)
        predictions = np.argmax(model.predict(img_reshaped))
        predicted_label = labels[predictions]
        self.assertIn(predicted_label, labels)

        

 
if __name__ == '__main__':
    unittest.main()
