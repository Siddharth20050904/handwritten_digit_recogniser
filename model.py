import numpy as np

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def d_sigmoid(s):
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def d_relu(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

class Layer:
    def __init__(self, input_dim, output_dim, activation='relu'):
        self.weights = np.random.randn(output_dim, input_dim) * np.sqrt(2. / input_dim)
        self.bias = np.zeros((output_dim, 1))
        self.activation = activation
        
    def forward(self, prev):
        self.prev = prev
        self.z = np.dot(self.weights, prev) + self.bias
        if self.activation=='relu':
            self.a = relu(self.z)
        elif self.activation=='sigmoid':
            self.a = sigmoid(self.z)
        elif self.activation == 'softmax':
            self.a = softmax(self.z)
        return self.a
    
    def backward(self, grad_output, learning_rate):
        if self.activation == 'relu':
            dz = grad_output * d_relu(self.a)
        elif self.activation=='sigmoid':
            dz = grad_output * d_sigmoid(self.a)
        elif self.activation == 'softmax':
            dz = grad_output
        dw = np.dot(dz, self.prev.T)
        db = dz
        
        grad_input = np.dot(self.weights.T, dz)
        
        self.weights -= learning_rate * dw
        self.bias -= learning_rate * db
        
        return grad_input
    
class NeuralNetwork:
    def __init__(self, layers=None):
        self.layers = layers if layers is not None else []
        
    def add(self, layer):
        self.layers.append(layer)
        
    def predict(self, x):
        result = self.forward(x)
        return np.argmax(result)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def save(self, filename):
        params = {}
        for i, layer in enumerate(self.layers):
            params[f"W{i}"] = layer.weights
            params[f"b{i}"] = layer.bias
            params[f"activation{i}"] = layer.activation
        np.savez(filename, **params)
    
    @classmethod
    def load(cls, filename):
        data = np.load(filename, allow_pickle=True)
        layers = []
        i = 0
        while f"W{i}" in data:
            w = data[f"W{i}"]
            b = data[f"b{i}"]
            act = str(data[f"activation{i}"])
            inp, out = w.shape[1], w.shape[0]
            layer = Layer(inp, out, activation=act)
            layer.weights = w
            layer.bias = b
            layers.append(layer)
            i += 1
        return cls(layers)