import torch

#prediction

# Define w = 2 and b = -1 for y = wx + b
print("Prediction: ", 2 * 1 - 1)
w = torch.tensor(2.0, requires_grad = True)
b = torch.tensor(-1.0, requires_grad = True)
print("Prediction: ", w * 1 + b)
# Function forward(x) for prediction

def forward(x):
    yhat = w * x + b
    print("Prediction calculated as: ", yhat)
    return yhat

# 𝑦̂ =−1+2𝑥
 
# 𝑦̂ =−1+2(1)

# Predict y = 2x - 1 at x = 1

x = torch.tensor([[1.0]])
yhat = forward(x)
print("The prediction returned: ", yhat)

# Create x Tensor and check the shape of x tensor

x = torch.tensor([[1.0], [2.0]])
print("The shape of x: ", x.shape)

# Make the prediction of y = 2x - 1 at x = [1, 2]

yhat = forward(x)
print("The prediction: ", yhat)

#CLASS Linear

# Import Class Linear
from torch.nn import Linear
print("Import Class Linear: ", torch.nn.Linear)

# Set random seed
print("Random seed: ", torch.manual_seed(1)) 

# Create Linear Regression Model, and print out the parameters

lr = Linear(in_features=1, out_features=1, bias=True)
print("lr Parameters w and b: ", list(lr.parameters()))


#𝑏=−0.44,𝑤=0.5153
 
# 𝑦̂ =−0.44+0.5153𝑥

print("lr dictionary: ",lr.state_dict())
print("lr keys: ",lr.state_dict().keys())
print("lr values: ",lr.state_dict().values())

print("lr weight:",lr.weight)
print("lr bias:",lr.bias)

# Make the prediction at x = [[1.0]]

x = torch.tensor([[1.0]])
print("lr prediction: ", lr(x))
yhat = lr(x)
print("lr prediction returned: \n", yhat)

#multiple predictions
# Create the prediction using linear model

x = torch.tensor([[1.0], [2.0]])
print("multiple lr prediction: ", lr(x))
yhat = lr(x)
print("The prediction returned: ", yhat)


# Practice: Use the linear regression model object lr to 
# make the prediction.

x = torch.tensor([[1.0],[2.0],[3.0]])
print("lr predictions returned \n: ", lr(x))

yhat = lr(x)

print("predictionS returned \n",yhat) 

#building Custom Modules

# Library for this section

from torch import nn

# Customize Linear Regression Class

class LR(nn.Module):
    
    # Constructor
    def __init__(self, input_size, output_size):
        
        # Inherit from parent
        super(LR, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    # Prediction function
    def forward(self, x):
        out = self.linear(x)
        return out
    

# Create the linear regression model. Print out the parameters.

lr = LR(1, 1)
print("The parameters: ", list(lr.parameters()))
print("Linear model: ", lr.linear)

# Try our customize linear regression model with single input

x = torch.tensor([[1.0]])
yhat = lr(x)
print("The prediction returned :", yhat)

# Try our customize linear regression model with multiple input

x = torch.tensor([[1.0], [2.0]])
yhat = lr(x)
print("The prediction returned: ", yhat)

print("Python dictionary: ", lr.state_dict())
print("keys: ",lr.state_dict().keys())
print("values: ",lr.state_dict().values())


print("weight: ", lr.linear.weight)
print("bias: ", lr.linear.bias)
print("weight: ", lr.state_dict()['linear.weight'])
print("bias: ", lr.state_dict()['linear.bias'])
print("weight: ", lr.state_dict()['linear.weight'][0][0])

print(">>>>>>>>>>>>>>>>>>>end of Line<<<<<<<<<<<<<<<<<<<<<")