""" Test the activation functions

Could use the unittest package for this, but instead I'm following the
pattern of the other tests.
"""

import torch
from torch.nn.functional import relu as relu_torch

from mlp import identity, relu, sigmoid

for size in [
    (1, 1),
    (2, 2),
    (5, 10),
]:
    tensor = torch.rand(size) * 2 - 1  # Uniform distribution over [-1, 1)
    identity_output = identity(tensor)
    relu_output = relu(tensor)
    sigmoid_output = sigmoid(tensor)

    print("Size", tuple(tensor.size()))
    print("  Identiy:", torch.allclose(identity_output, tensor))
    print("  ReLU:", torch.allclose(relu_output, relu_torch(tensor)))
    print("  Sigmoid:", torch.allclose(sigmoid_output, torch.sigmoid(tensor)))
