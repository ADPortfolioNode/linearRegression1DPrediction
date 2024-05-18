import torch
import numpy as np

import matplotlib.pyplot as plt

# The class for plotting
class plot_diagram():
    # Constructor
    def __init__(self, X, Y, w, stop, go=False):
        start = w.data
        self.error = []
        self.parameter = []
        self.X = X.numpy()
        self.Y = Y.numpy()
        self.parameter_values = torch.arange(start, stop)
        self.Loss_function = [criterion(forward(X), Y) for _ in self.parameter_values]
        w.data = start

    # Executor
    def __call__(self, Yhat, w, error, n):
        self.error.append(error)
        self.parameter.append(w.data)
        plt.subplot(212)
        plt.plot(self.X, Yhat.detach().numpy())
        plt.plot(self.X, self.Y, 'ro')
        plt.xlabel("A")
        plt.ylim(-20, 20)
        plt.subplot(211)
        plt.title("Data Space (top) Estimated Line (bottom) Iteration " + str(n))
        # Convert lists to PyTorch tensors
        
        parameter_values_tensor = torch.tensor(self.parameter_values)
        loss_function_tensor = torch.tensor(self.Loss_function)

        # Plot using the tensors
        plt.plot(parameter_values_tensor.numpy(), loss_function_tensor.numpy())

        plt.plot(self.parameter, self.error, 'ro')
        plt.xlabel("B")
        plt.figure()

    # Destructor
    def __del__(self):
        plt.close('all')


# Create the f(X) with a slope of -3
X = torch.arange(-3, 3, 0.1).view(-1, 1)
f = -3 * X

# Plot the line with blue
plt.plot(X.numpy(), f.numpy(), label='f')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# Add some noise to f(X) and save it in Y
Y = f + 0.1 * torch.randn(X.size())

# Plot the data points
plt.plot(X.numpy(), Y.numpy(), 'rx', label='Y')
plt.plot(X.numpy(), f.numpy(), label='f')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# CREATE MODEL AND COST FUNCTION

# First, define the forward function 𝑦=𝑤∗𝑥
# . (We will add the bias next.)

# Create forward function for prediction
def forward(x):
    return w * x

# Create the MSE function to evaluate the result.
def criterion(yhat, y):
    return torch.mean((yhat - y) ** 2)

# Create Learning Rate and an empty list to record the loss for each iteration
lr = 0.1
LOSS = []

w = torch.tensor(-10.0, requires_grad=True)

gradient_plot = plot_diagram(X, Y, w, stop=5)

# Define a function to train the model
def train_model(iter):
    for epoch in range(iter):
        # Make the prediction
        Yhat = forward(X)

        # Calculate the loss
        loss = criterion(Yhat, Y)

        # Plot the diagram
        gradient_plot(Yhat, w, loss.item(), epoch)

        # Store the loss into list
        LOSS.append(loss.item())

        # Backward pass: compute gradient of the loss with respect to all the learnable parameters
        loss.backward()

        # Update parameters
        w.data = w.data - lr * w.grad.data

        # Zero the gradients before running the backward pass
        w.grad.data.zero_()

# Train the model for 4 iterations
train_model(4)

w = torch.tensor(-15.0, requires_grad=True)
LOSS2 = []

gradient_plot1 = plot_diagram(X, Y, w, stop=5)

def my_train_model(iter):
    for epoch in range(iter):
        Yhat = forward(X)
        loss = criterion(Yhat, Y)
        gradient_plot1(Yhat, w, loss.item(), epoch)
        LOSS2.append(loss.item())
        loss.backward()
        w.data = w.data - lr * w.grad.data
        w.grad.data.zero_()

my_train_model(4)

gradient_plot = plot_diagram(X, Y, w, stop=15)

plt.plot(LOSS, label="LOSS")
plt.plot(LOSS2, label="LOSS2")
plt.tight_layout()
plt.xlabel("Epoch/Iterations")
plt.ylabel("Cost")
plt.legend()
plt.show()


print(">>>>>>>>>>>>>>>>>>>>>>>>End of Line<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")