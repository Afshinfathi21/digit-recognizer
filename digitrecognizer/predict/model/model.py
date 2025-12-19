import numpy as np
from matplotlib import pyplot as plt
import struct
from tqdm import trange
# def load_mnist_images(path):
#     with open(path, 'rb') as f:
#         magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
#         if magic != 2051:
#             raise ValueError("Invalid magic number for MNIST image file")
#         images = np.frombuffer(f.read(), dtype=np.uint8)
#         images = images.reshape(num, rows, cols)
#         return images

# def load_mnist_labels(path):
#     with open(path, 'rb') as f:
#         magic, num = struct.unpack(">II", f.read(8))
#         if magic != 2049:
#             raise ValueError("Invalid magic number for MNIST label file")
#         labels = np.frombuffer(f.read(), dtype=np.uint8)
#         return labels
# X_train = load_mnist_images("train-images.idx3-ubyte")
# y_train = load_mnist_labels("train-labels.idx1-ubyte")

# X_test = load_mnist_images("t10k-images.idx3-ubyte")
# y_test = load_mnist_labels("t10k-labels.idx1-ubyte")

# print(X_train.shape)  # (60000, 28, 28)
# print(y_train.shape)  # (60000,)


# X_train = X_train.reshape(-1,28*28)
# X_test = X_test.reshape(-1,28*28)



# X_train = X_train.astype('float32') / 255.0
# X_test = X_test.astype('float32') / 255.0

# def onehot_encoder(labels,length=10):
#     labels = np.array(labels)
#     one_hot = np.zeros((labels.shape[0],length))
#     one_hot[np.arange(labels.shape[0]),labels] = 1
#     return one_hot
# y_train = onehot_encoder(y_train,10)

# print(X_train.shape,y_train.shape)



# def plot_training(losses):
#     # Plot the loss
#     plt.plot(losses)
#     plt.title("Training loss")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.show()


class Layer:
    def __init__(self):
        self.inp = None
        self.out = None

    def __call__(self, inp: np.ndarray) -> np.ndarray:
        return self.forward(inp)

    def forward(self, inp: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, up_grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def step(self, lr: float) -> None:
        pass
    

class Linear(Layer):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.w = 0.1 * np.random.randn(in_dim, out_dim)
        self.b = np.zeros((1, out_dim))
        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)

    def forward(self, inp: np.ndarray) -> np.ndarray:
        """Perform the linear transformation: output = inp * W + b"""
        self.inp = inp
        self.out = np.dot(inp, self.w) + self.b
        return self.out

    def backward(self, up_grad: np.ndarray) -> np.ndarray:
        """Backpropagate the gradients through this layer."""

        self.dw = np.dot(self.inp.T, up_grad)  
        self.db = np.sum(up_grad, axis=0, keepdims=True)  

        down_grad = np.dot(up_grad, self.w.T)
        return down_grad

    def step(self, lr: float) -> None:
        """Update the weights and biases using the gradients."""
        self.w -= lr * self.dw
        self.b -= lr * self.db

class ReLU(Layer):
    def forward(self, inp: np.ndarray) -> np.ndarray:
        """ReLU Activation: f(x) = max(0, x)"""
        self.inp = inp
        self.out = np.maximum(0, inp)
        return self.out

    def backward(self, up_grad: np.ndarray) -> np.ndarray:
        """Backward pass for ReLU: derivative is 1 where input > 0, else 0."""
        down_grad = up_grad * (self.inp > 0)
        return down_grad

class Softmax(Layer):
    def forward(self, inp: np.ndarray) -> np.ndarray:
        """Softmax Activation: f(x) = exp(x) / sum(exp(x))"""

        exp_values = np.exp(inp - np.max(inp, axis=1, keepdims=True))
        self.out = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.out

    def backward(self, up_grad: np.ndarray) -> np.ndarray:
        """Backward pass for Softmax using the Jacobian matrix."""
        down_grad = np.empty_like(up_grad)
        for i in range(up_grad.shape[0]):
            single_output = self.out[i].reshape(-1, 1)
            jacobian = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            down_grad[i] = np.dot(jacobian, up_grad[i])
        return down_grad

class Loss:
    def __init__(self):
        self.prediction = None
        self.target = None
        self.loss = None

    def __call__(self, prediction: np.ndarray, target: np.ndarray) -> float:
        return self.forward(prediction, target)

    def forward(self, prediction: np.ndarray, target: np.ndarray) -> float:
        raise NotImplementedError

    def backward(self) -> np.ndarray:
        raise NotImplementedError
    
class CrossEntropy(Loss):
    def forward(self, prediction: np.ndarray, target: np.ndarray) -> float:
        """Cross-Entropy Loss for classification."""
        self.prediction = prediction
        self.target = target

        clipped_pred = np.clip(prediction, 1e-12, 1.0)
        self.loss = -np.mean(np.sum(target * np.log(clipped_pred), axis=1))
        return self.loss

    def backward(self) -> np.ndarray:
        """Gradient of Cross-Entropy Loss."""
        grad = -self.target / self.prediction / self.target.shape[0]
        return grad

class MLP:
    def __init__(self, layers: list[Layer], loss_fn: Loss, lr: float) -> None:
        """
        Multi-Layer Perceptron (MLP) class.
        Arguments:
        - layers: List of layers (e.g., Linear, ReLU, etc.).
        - loss_fn: Loss function object (e.g., CrossEntropy, MSE).
        - lr: Learning rate.
        """
        self.layers = layers
        self.loss_fn = loss_fn
        self.lr = lr

    def __call__(self, inp: np.ndarray) -> np.ndarray:
        """Makes the model callable, equivalent to forward pass."""
        return self.forward(inp)
    
    def save(self, path: str) -> None:
        """
        Save model weights to disk.
        Only Linear layers have parameters.
        """
        params = {}
        linear_idx = 0

        for layer in self.layers:
            if isinstance(layer, Linear):
                params[f"W{linear_idx}"] = layer.w
                params[f"b{linear_idx}"] = layer.b
                linear_idx += 1

        np.savez(path, **params)

    def load(self, path: str) -> None:
        """
        Load model weights from disk.
        """
        data = np.load(path)
        linear_idx = 0

        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.w = data[f"W{linear_idx}"]
                layer.b = data[f"b{linear_idx}"]
                linear_idx += 1

    def forward(self, inp: np.ndarray) -> np.ndarray:
        """Pass input through each layer sequentially."""
        for layer in self.layers:
            inp = layer.forward(inp)
        return inp

    def loss(self, prediction: np.ndarray, target: np.ndarray) -> float:
        """Calculate the loss."""
        return self.loss_fn(prediction, target)

    def backward(self) -> None:
        """Perform backpropagation by propagating the gradient backwards through the layers."""
        up_grad = self.loss_fn.backward()
        for layer in reversed(self.layers):
            up_grad = layer.backward(up_grad)

    def update(self) -> None:
        """Update the parameters of each layer using the gradients and the learning rate."""
        for layer in self.layers:
            layer.step(self.lr)

    def train(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int, batch_size: int) -> np.ndarray:
        """Train the MLP over the given dataset for a number of epochs."""
        losses = np.empty(epochs)
        for epoch in (pbar := trange(epochs)):
            running_loss = 0.0
            for i in range(0, len(x_train), batch_size):
                x_batch = x_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]


                prediction = self.forward(x_batch)

                running_loss += self.loss(prediction, y_batch) * batch_size

                self.backward()


                self.update()

            running_loss /= len(x_train)
            pbar.set_description(f"Loss: {running_loss:.3f}")
            losses[epoch] = running_loss

        return losses
    
layers = [Linear(784, 128),
          ReLU(),
          Linear(128, 128),
          ReLU(),
          Linear(128,10),
          Softmax()]

model = MLP(layers, CrossEntropy(), lr=0.001)

# losses = model.train(X_train, y_train, epochs=30, batch_size=64)
# model.save("mnist_mlp_weights.npz")
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "mnist_mlp_weights.npz")
model.load(WEIGHTS_PATH)
# plot_training(losses)

# y_prediction = np.argmax(model(X_test), axis=1)
# acc = 100 * np.mean(y_prediction == y_test)
# print(f'Test accuracy with {len(y_train)} training examples on {len(y_test)} test samples is {acc:.2f}%')



def predict(input):
    predict = np.argmax(model(input))
    return predict
