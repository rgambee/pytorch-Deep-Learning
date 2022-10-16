import torch


def identity(x):
    return x


def relu(x):
    x = x.clone()
    x[x < 0] = 0
    return x


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


ACTIVATION_FUNCTIONS = {
    "identity": identity,
    "relu": relu,
    "sigmoid": sigmoid,
}


def identity_grad(x):
    return torch.ones_like(x)


def relu_grad(x):
    x_prime = torch.ones_like(x)
    x_prime[x < 0] = 0
    return x_prime


def sigmoid_grad(x):
    return torch.exp(-x) / torch.pow(1 + torch.exp(-x), 2)


ACTIVATION_GRADIENTS = {
    "identity": identity_grad,
    "relu": relu_grad,
    "sigmoid": sigmoid_grad,
}

class MLP:
    def __init__(
        self,
        linear_1_in_features,
        linear_1_out_features,
        f_function,
        linear_2_in_features,
        linear_2_out_features,
        g_function
    ):
        """
        Args:
            linear_1_in_features: the in features of first linear layer
            linear_1_out_features: the out features of first linear layer
            linear_2_in_features: the in features of second linear layer
            linear_2_out_features: the out features of second linear layer
            f_function: string for the f function: relu | sigmoid | identity
            g_function: string for the g function: relu | sigmoid | identity
        """
        self.f_function = ACTIVATION_FUNCTIONS[f_function]
        self.f_prime = ACTIVATION_GRADIENTS[f_function]
        self.g_function = ACTIVATION_FUNCTIONS[g_function]
        self.g_prime = ACTIVATION_GRADIENTS[g_function]

        self.parameters = dict(
            W1 = torch.randn(linear_1_out_features, linear_1_in_features),
            b1 = torch.randn(linear_1_out_features),
            W2 = torch.randn(linear_2_out_features, linear_2_in_features),
            b2 = torch.randn(linear_2_out_features),
        )
        self.grads = dict(
            dJdW1 = torch.zeros(linear_1_out_features, linear_1_in_features),
            dJdb1 = torch.zeros(linear_1_out_features),
            dJdW2 = torch.zeros(linear_2_out_features, linear_2_in_features),
            dJdb2 = torch.zeros(linear_2_out_features),
        )

        # put all the cache value you need in self.cache
        self.cache = dict()

    def forward(self, x):
        """
        Args:
            x: tensor shape (batch_size, linear_1_in_features)
        """
        batch_size = x.size()[0]
        self.cache["x"] = x
        self.cache["z1"] = (
            torch.mm(x, self.parameters["W1"].T)
            + self.parameters["b1"].repeat((batch_size, 1))
        )
        self.cache["z2"] = self.f_function(self.cache["z1"])
        self.cache["z3"] = (
            torch.mm(self.cache["z2"], self.parameters["W2"].T)
            + self.parameters["b2"].repeat((batch_size, 1))
        )
        self.cache["z4"] = self.g_function(self.cache["z3"])
        return self.cache["z4"]
    
    def backward(self, dJdy_hat):
        """
        Args:
            dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
        """
        # y_hat = g(W2 * f(W1 * x + b1) + b2)
        # dJ/db1 = dJ/dy_hat * dy_hat/db1

        # y_hat = g(z3)             y_hat/dz3 = g'(z3)
        # z3 = W2 * z2 + b2         dz3/dz2   = W2
        # z2 = f(z1)                dz2/dz1   = f'(z1)
        # z1 = W1 * x + b1          dz1/dx    = W1

        # TODO: try element-wise multiplication of 1D vectors,
        # not expand, matmul, truncate

        batch_size = dJdy_hat.size(0)

        # Assign to self.grads["..."][:] to avoid changing the shape
        self.grads["dJdb2"][:] = torch.mm(
            self.g_prime(self.cache["z3"]).T, dJdy_hat
        )[0, :]

        # dJdb2 = torch.ones((self.grads["dJdW2"].size(0), self.cache["z2"].size(0)))
        # dJdb2[:, 0] = self.grads["dJdb2"]
        self.grads["dJdW2"][:] = torch.mm(
            self.grads["dJdb2"].repeat((batch_size, 1)).T,
            self.cache["z2"]
        )

        # print(self.grads["dJdb2"].size())
        # print(self.cache["z1"].size())
        # print(self.parameters["W2"].size())
        # print(self.grads["dJdb1"].size())
        self.grads["dJdb1"][:] = torch.mm(
            self.parameters["W2"].T,
            torch.mm(
                self.grads["dJdb2"].repeat((batch_size, 1)).T,
                self.f_prime(self.cache["z1"])
            ),
        # Why does dividing by the batch size here give the right answer?
        # Is it because the b2 gradient was repeated?
        )[:, 0] / batch_size
        
        self.grads["dJdW1"][:] = torch.mm(
            self.grads["dJdb1"].repeat((batch_size, 1)).T,
            self.cache["x"]
        )
    
    def clear_grad_and_cache(self):
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()


def mse_loss(y, y_hat):
    """
    Args:
        y: the label tensor (batch_size, linear_2_out_features)
        y_hat: the prediction tensor (batch_size, linear_2_out_features)

    Return:
        J: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    J = torch.pow(y_hat - y, 2).mean()
    dJdy_hat = 2 * (y_hat - y) / y.numel()
    return J, dJdy_hat


def bce_loss(y, y_hat):
    """
    Args:
        y_hat: the prediction tensor
        y: the label tensor
        
    Return:
        loss: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    J = -(y * torch.log(y_hat) + (1 - y) * torch.log(1 - y_hat)).mean()
    dJdy_hat = -y / y_hat + (1 - y) / (1 - y_hat)
    return J, dJdy_hat
