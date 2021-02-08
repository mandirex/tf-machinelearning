
# import tensorflow package
import tensorflow as tf 

if __name__ == "__main__":

    # Download mnist dataset (the hello world data set of machine learning)
    mnist = tf.keras.datasets.mnist

    # x in, y out. Split into separate test and train data.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Normalize pixel data from 0-255 to 0.0 - 1.0
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)), # We flatten the 2D pixel array into one long array
        tf.keras.layers.Dense(128, activation="relu"), # Fully connected layers using the relu activation function
        tf.keras.layers.Dropout(0.2), # randomly sets input units to 0 during training, which helps fighting overfitting.
        tf.keras.layers.Dense(10) # 10 output neurons.
    ])


    print("testing")