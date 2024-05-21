
import torch
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits import mplot3d

# The class for plot the diagram

class plot_error_surfaces(object):
    
    # Constructor
    #print("plotting srfaces")
    def __init__(self, w_range, b_range, X, Y, n_samples = 30, go = True):
        W = np.linspace(-w_range, w_range, n_samples)
        B = np.linspace(-b_range, b_range, n_samples)
        w, b = np.meshgrid(W, B)    
        Z = np.zeros((30, 30))
        count1 = 0
        self.y = Y.numpy()
        self.x = X.numpy()
        for w1, b1 in zip(w, b):
            count2 = 0
            for w2, b2 in zip(w1, b1):
                Z[count1, count2] = np.mean((self.y - w2 * self.x + b2) ** 2)
                count2 += 1
            count1 += 1
        self.Z = Z
        self.w = w
        self.b = b
        self.W = []
        self.B = []
        self.LOSS = []
        self.n = 0
        if go == True:
            plt.figure()
            plt.figure(figsize = (7.5, 5))
            plt.axes(projection = '3d').plot_surface(self.w, self.b, self.Z, rstride = 1, cstride = 1,cmap = 'viridis', edgecolor = 'none')
            plt.title('Loss Surface')
            plt.xlabel('w')
            plt.ylabel('b')
            plt.show()
            plt.figure()
            plt.title('Loss Surface Contour')
            plt.xlabel('w')
            plt.ylabel('b')
            plt.contour(self.w, self.b, self.Z)
            plt.show()
    
    # Setter
    def set_para_loss(self, W, B, loss):
        #print("setting parameters")
        self.n = self.n + 1
        self.W.append(W)
        self.B.append(B)
        self.LOSS.append(loss)
    
    # Plot diagram
    def final_plot(self): 
        ax = plt.axes(projection = '3d')
        #print("plotting final plot")
        ax.plot_wireframe(self.w, self.b, self.Z)
        ax.scatter(self.W, self.B, self.LOSS, c = 'r', marker = 'x', s = 200, alpha = 1)
        plt.figure()
        plt.contour(self.w, self.b, self.Z)
        plt.scatter(self.W, self.B, c = 'r', marker = 'x')
        plt.xlabel('w')
        plt.ylabel('b')
        plt.title('Loss Surface Contour')
        plt.show()
    
    # Plot diagram
    def plot_ps(self):
        #print("plotting ps")
        plt.subplot(121)
        plt.ylim
        plt.plot(self.x, self.y, 'ro', label = "training points")
        plt.plot(self.x, self.W[-1] * self.x + self.B[-1], label = "estimated line")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.ylim((-10, 15))
        plt.title('Data Space Iteration: ' + str(self.n))
        plt.subplot(122)
        plt.contour(self.w, self.b, self.Z)
        plt.scatter(self.W, self.B, c = 'r', marker = 'x')
        plt.title('Loss Surface Contour Iteration' + str(self.n))
        plt.xlabel('w')
        plt.ylabel('b')
        plt.show()
        
# Set random seed
#print("setting random seed to 1")
torch.manual_seed(1)

# Setup the actual data and simulated data

X = torch.arange(-3, 3, 0.1).view(-1, 1)
f = 1 * X - 1
Y = f + 0.1 * torch.randn(X.size())
#print("setting up data and simulated data")
# Plot out the data dots and line
#print("plotting data")
plt.plot(X.numpy(), Y.numpy(), 'rx', label = 'y')
plt.plot(X.numpy(), f.numpy(), label = 'f')
plt.xlabel('x')
plt.ylabel('y')
plt.title('data points')
plt.legend()
plt.show()

# Define the forward function

def forward(x):
    #print("prediction made by forward function")
    return w * x + b

# Define the MSE Loss function

def criterion(yhat, y):
    #print("loss calculated by criterion function")
    return torch.mean((yhat - y) ** 2)

# Create plot_error_surfaces for viewing the data

get_surface = plot_error_surfaces(15, 13, X, Y, 30)
#print("plotting error surfaces")
###TRAIN THE MODEL:BATCH GRADIENT DESCENT#########################################################################################################

# Define the parameters w, b for y = wx + b

w = torch.tensor(-15.0, requires_grad = True)
b = torch.tensor(-10.0, requires_grad = True)
#print("parameters w and b defined")
# Define learning rate and create an empty list for containing the loss for each iteration.

lr = 0.1
LOSS_BGD = []

# The function for training the model

def train_model(iter):
    #print("training model ",iter," times")
    # Loop
    for epoch in range(iter):
        
        # make a prediction
        Yhat = forward(X)
        
        # calculate the loss 
        loss = criterion(Yhat, Y)

        # Section for plotting
        get_surface.set_para_loss(w.data.tolist(), b.data.tolist(), loss.tolist())
        get_surface.plot_ps()
            
        # store the loss in the list LOSS_BGD
        LOSS_BGD.append(loss)
        
        # backward pass: compute gradient of the loss with respect to all the learnable parameters
        loss.backward()
        
        # update parameters slope and bias
        w.data = w.data - lr * w.grad.data
        b.data = b.data - lr * b.grad.data
        
        # zero the gradients before running the backward pass
        w.grad.data.zero_()
        b.grad.data.zero_()
    #print("training completed")
        
# Train the model with 10 iterations
#print("training model with 10 iterations")
train_model(10)

######TRAIN THE MODEL STOCHASTIC GREADIENT#################################################################        

# The function for training the model

LOSS_SGD = []
w = torch.tensor(-15.0, requires_grad = True)
b = torch.tensor(-10.0, requires_grad = True)

def train_model_SGD(iter):
    #print("training model with SGD ",iter," times")
    # Loop
    for epoch in range(iter):
        
        # SGD is an approximation of out true total loss/cost, in this line of code we calculate our true loss/cost and store it
        Yhat = forward(X)
        #print("SGD approximation of total loss calculated")
        # store the loss 
        LOSS_SGD.append(criterion(Yhat, Y).tolist())
        
        for x, y in zip(X, Y):
            
            # make a pridiction
            yhat = forward(x)
        
            # calculate the loss 
            loss = criterion(yhat, y)

            # Section for plotting
            get_surface.set_para_loss(w.data.tolist(), b.data.tolist(), loss.tolist())
        
            # backward pass: compute gradient of the loss with respect to all the learnable parameters
            loss.backward()
        
            # update parameters slope and bias
            w.data = w.data - lr * w.grad.data
            b.data = b.data - lr * b.grad.data

            # zero the gradients before running the backward pass
            w.grad.data.zero_()
            b.grad.data.zero_()
            
        #plot surface and data space after each epoch  
        #print("plotting surface and data space after each epoch")  
        get_surface.plot_ps()
# Train the model with 10 iterations
#print("training model with SGD 10 times")
train_model_SGD(10)

# Plot out the LOSS_BGD and LOSS_SGD
LOSS_BGD= [ loss.detach().numpy() for loss in LOSS_BGD]
plt.plot(LOSS_BGD,label = "Batch Gradient Descent")
plt.plot(LOSS_SGD,label = "Stochastic Gradient Descent")
plt.xlabel('epoch')
plt.ylabel('Cost/ total loss')
plt.legend()
plt.show()



#print("training with data loader")
# Import the library for DataLoader

from torch.utils.data import Dataset, DataLoader

# Dataset Class

class Data(Dataset):
    #print("dataset class created")
    # Constructor
    def __init__(self):
        self.x = torch.arange(-3, 3, 0.1).view(-1, 1)
        self.y = 1 * self.x - 1
        self.len = self.x.shape[0]
        
    # Getter
    #print("getter created")
    def __getitem__(self,index):    
        return self.x[index], self.y[index]
    
    # Return the length
    #print("length getter created")
    def __len__(self):
        return self.len
    
    
    
# Create the dataset and check the length

dataset = Data()
#print("The length of dataset: ", len(dataset))

# #print the first point

x, y = dataset[0]
#print("(", x, ", ", y, ")")

# #print the first 3 point

x, y = dataset[0:3]
#print("The first 3 x: ", x)
#print("The first 3 y: ", y)


# Create plot_error_surfaces for viewing the data

get_surface = plot_error_surfaces(15, 13, X, Y, 30, go = False)
#print
# Create DataLoader
#print
trainloader = DataLoader(dataset = dataset, batch_size = 1)

# The function for training the model

w = torch.tensor(-15.0,requires_grad=True)
b = torch.tensor(-10.0,requires_grad=True)
LOSS_Loader = []

def train_model_DataLoader(epochs):
    #print("training model dataloader ",epochs," times")
    # Loop
    for epoch in range(epochs):
        
        # SGD is an approximation of out true total loss/cost, in this line of code we calculate our true loss/cost and store it
        Yhat = forward(X)
        #print("SGD approximation of total loss calculated")
        # store the loss 
        LOSS_Loader.append(criterion(Yhat, Y).tolist())
        #print("loss stored")
        for x, y in trainloader:
            
            # make a prediction
            yhat = forward(x)
            
            # calculate the loss
            loss = criterion(yhat, y)
            
            # Section for plotting
            get_surface.set_para_loss(w.data.tolist(), b.data.tolist(), loss.tolist())
            
            # Backward pass: compute gradient of the loss with respect to all the learnable parameters
            loss.backward()
            #print("loss backward calculated")
            # Updata parameters slope
            w.data = w.data - lr * w.grad.data
            b.data = b.data - lr* b.grad.data
            #print("parameters updated")
            # Clear gradients 
            w.grad.data.zero_()
            b.grad.data.zero_()
            #print("gradients cleared")
            
        #plot surface and data space after each epoch    
        get_surface.plot_ps()
        
        
# Run 10 iterations
#print("training model with dataloader 10 times")
train_model_DataLoader(10)


# Plot the LOSS_BGD and LOSS_Loader
#print("plotting LOSS_BGD and LOSS_Loader")
plt.plot(LOSS_BGD,label="Batch Gradient Descent")
plt.plot(LOSS_Loader,label="Stochastic Gradient Descent with DataLoader")
plt.xlabel('epoch')
plt.ylabel('Cost/ total loss')
plt.legend()
plt.show()
######################################################################
# Practice: Use SGD with trainloader to train model and store the total loss in LOSS

LOSS = []
w = torch.tensor(-12.0, requires_grad = True)
b = torch.tensor(-10.0, requires_grad = True)
print("parameters w and b defined")
def my_train_model(epochs):
    print("training model with SGD and DataLoader ",epochs," times")
    for epochs in range(epochs):
        yhat = forward(x)
        loss = criterion(yhat,y)
        get_surface.set_para_loss(w.data.tolist(), b.data.tolist(), loss.tolist())
        loss.backward()
        w.data = w.data =lr * w.grad.data
        b.data = b.data - lr * b.grad.data
        w.grad.data.zero_()
        b.grad.data.zero_()
    get_surface.plot_ps()
    print("SGD with DataLoader training completed")
    
# Train the model with 10 iterations
print("training model with SGD and DataLoader 10 times")
my_train_model(10)                            



#print(">>>>>>>>>>>>>>>>>>>>>>>>End of Line<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")