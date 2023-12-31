import torch
import torch.nn as nn
# from parameters import *

# Define the network architecture




model = nn.Sequential(
    nn.Conv2d(in_channels=4, out_channels=6, kernel_size=7, stride=3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(in_channels=6, out_channels=12, kernel_size=4),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),

)

# Dummy input tensor with the shape (batch_size, channels, height, width)
dummy_input = torch.randn(1, 4, 96, 96)

# Pass the dummy input through the model to get the output tensor
output_tensor = model(dummy_input)



# Print the shape of the output tensor
print("Output Tensor Shape:", output_tensor.shape)
print(torch.argmax(output_tensor))
