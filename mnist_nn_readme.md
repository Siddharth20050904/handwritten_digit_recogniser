# Neural Network from Scratch - MNIST Digit Classification

A custom implementation of a neural network built from scratch using NumPy for MNIST digit classification. This project demonstrates the fundamentals of deep learning by implementing forward propagation, backpropagation, and various activation functions without relying on high-level deep learning frameworks.

## Features

- **Custom Neural Network Implementation**: Built entirely with NumPy
- **Multiple Activation Functions**: ReLU, Sigmoid, and Softmax
- **Flexible Architecture**: Easy to add/remove layers
- **Model Persistence**: Save and load trained models
- **MNIST Dataset**: Handwritten digit classification (0-9)
- **Batch Processing**: Efficient training with batched data
- **Cross-Entropy Loss**: Appropriate loss function for multi-class classification

## Architecture

The implemented neural network consists of:
- **Input Layer**: 784 neurons (28×28 flattened MNIST images)
- **Hidden Layer 1**: 64 neurons with ReLU activation
- **Hidden Layer 2**: 32 neurons with ReLU activation
- **Output Layer**: 10 neurons with Softmax activation (for 10 digit classes)

## Project Structure

```
project/
│
├── node.py                 # Core neural network implementation
├── train_model.ipynb          # Training script
├── test_model.py           # Testing/evaluation script
├── model.npz              # Saved trained model (generated after training)
└── README.md              # This file
```

## Requirements

```bash
numpy
tensorflow  # Only for MNIST dataset loading and batching
keras      # For MNIST dataset
```

## Installation

1. Clone or download the project files
2. Install required dependencies:
   ```bash
   pip install numpy tensorflow keras
   ```

## Usage

### Training the Model

Run the training script to train the neural network on MNIST data:

```python
# This will train for 20 epochs and save the model as 'model.npz'
python train_model.ipynb
```

Expected output:
```
Training images shape: (60000, 28, 28)
Training labels shape: (60000,)
Testing images shape: (10000, 28, 28)
Testing labels shape: (10000,)
Average Cost for epoch 1: 2.845
Average Cost for epoch 2: 1.923
...
Average Cost for epoch 20: 0.234
```

### Testing the Model

Evaluate the trained model on test data:

```python
python test_model.py
```

Expected output:
```
Accuracy: 96.67
```

### Using the Neural Network Class

```python
from node import Layer, NeuralNetwork
import numpy as np

# Create a custom network
model = NeuralNetwork()
model.add(Layer(784, 128, activation='relu'))
model.add(Layer(128, 64, activation='relu'))
model.add(Layer(64, 10, activation='softmax'))

# Make predictions
prediction = model.predict(input_data)

# Save model
model.save("my_model.npz")

# Load model
loaded_model = NeuralNetwork.load("my_model.npz")
```

## Implementation Details

### Core Components

#### Layer Class
- **Initialization**: Random weight initialization using He initialization for ReLU layers
- **Forward Pass**: Computes weighted sum + bias, applies activation function
- **Backward Pass**: Computes gradients and updates weights using gradient descent

#### Activation Functions
- **ReLU**: `max(0, x)` - Used in hidden layers
- **Sigmoid**: `1/(1 + e^(-x))` - Alternative activation function
- **Softmax**: Normalized exponential function - Used in output layer for probability distribution

#### Loss Function
- **Cross-Entropy Loss**: Appropriate for multi-class classification
- **Derivative**: Used for backpropagation

### Training Process

1. **Data Preprocessing**: 
   - Reshape 28×28 images to 784-dimensional vectors
   - Normalize pixel values to [0,1] range
   - Convert labels to one-hot encoding

2. **Forward Propagation**:
   - Pass input through each layer
   - Apply activation functions
   - Compute final predictions

3. **Loss Calculation**:
   - Compare predictions with true labels using cross-entropy

4. **Backpropagation**:
   - Compute gradients layer by layer
   - Update weights and biases using gradient descent

5. **Batch Processing**:
   - Process data in batches of 100 samples
   - Average gradients over batch for stable training

## Hyperparameters

- **Learning Rate**: 0.01
- **Batch Size**: 100
- **Epochs**: 20
- **Architecture**: 784 → 64 → 32 → 10

## Performance

The model typically achieves:
- **Training**: Converges after ~15-20 epochs
- **Test Accuracy**: 95-98% on MNIST test set
- **Training Time**: ~2-5 minutes on modern hardware

## Customization

### Changing Architecture
```python
# Example: Deeper network
layer1 = Layer(784, 128, activation='relu')
layer2 = Layer(128, 64, activation='relu')
layer3 = Layer(64, 32, activation='relu')
layer4 = Layer(32, 10, activation='softmax')
```

### Adjusting Hyperparameters
```python
# In training loop
learning_rate = 0.001  # Lower learning rate
epochs = 50           # More epochs
batch_size = 32       # Smaller batches
```

### Different Activation Functions
```python
layer = Layer(64, 32, activation='sigmoid')  # Use sigmoid instead of ReLU
```

## Limitations

- **No GPU Support**: Pure NumPy implementation (CPU only)
- **Basic Optimization**: Only supports vanilla gradient descent
- **Limited Regularization**: No dropout, batch normalization, etc.
- **Fixed Architecture**: Manual layer definition required

## Future Improvements

- [ ] Add different optimizers (Adam, RMSprop)
- [ ] Implement regularization techniques
- [ ] Add support for different loss functions
- [ ] Include data augmentation
- [ ] Add learning rate scheduling
- [ ] Implement early stopping

## Educational Value

This project demonstrates:
- **Fundamental Neural Network Concepts**: Forward/backward propagation
- **Gradient Descent Implementation**: Manual weight updates
- **Activation Functions**: Different non-linear transformations
- **Loss Functions**: Cross-entropy for classification
- **Batch Processing**: Efficient training techniques
- **Model Persistence**: Saving/loading trained models

## License

This project is for educational purposes. Feel free to modify and experiment with the code.

## Contributing

Contributions are welcome! Areas for improvement:
- Code optimization
- Additional activation functions
- Better initialization schemes
- More sophisticated optimizers
- Documentation improvements