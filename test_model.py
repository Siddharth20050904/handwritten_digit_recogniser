from keras.datasets import mnist
import numpy as np
from model import NeuralNetwork
import tensorflow as tf

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape(train_images.shape[0],-1)/255
train_labels = train_labels.reshape(-1, 1)
training_data = np.concatenate((train_images, train_labels), axis=1)

dataset = tf.data.Dataset.from_tensor_slices(training_data)
batched_data = dataset.batch(batch_size=100)

model = NeuralNetwork.load("model.npz")

correct = 0
total = 0

for i in range(test_images.shape[0]):
    y_pred = model.predict(np.reshape(test_images[i], (-1,1)))
    if y_pred == test_labels[i]: 
        correct+=1
    total+=1

print(f"Accuracy: {correct * 100 / total}")
    
