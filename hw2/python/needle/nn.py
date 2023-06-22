"""The module.
"""
from re import X
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []




class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        # y=xA'+b
        # A'.shape=(Hin,Hout) = weights
        self.weight = init.kaiming_uniform(in_features, out_features)
        if bias:
          self.bias = init.kaiming_uniform(out_features, 1).reshape((1, out_features))
        else:
          self.bias = init.zeros(out_features, 1).reshape((1, out_features))
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # X (N,Hin) weight (Hin,Hout) b (N,Hout)
        # Note: explicitly broadcast the bias term to the correct shape
        return X @ self.weight + \
          self.bias.broadcast_to((X.shape[0], self.out_features))
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        output = x
        for module in self.modules:
          output = module(output)
        return output
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        # logits (ndl.Tensor[np.float32]): 2D Tensor of shape
        #     (batch_size, num_classes), containing the logit predictions for
        #     each class.
        # y (ndl.Tensor[np.int8]): 1D Tensor of shape (batch_size, )
        #     containing the true label of each example.

        softmax = ops.logsumexp(logits,axes=1)
        batch_size = logits.shape[0]
        num_classes = logits.shape[1]

        one_hot = init.one_hot(num_classes, y)

        z_y = (logits * one_hot).sum(axes=1)
        loss = softmax - z_y
        total_loss = loss.sum()
        return total_loss / batch_size
        ### END YOUR SOLUTION



class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim))
        self.bias = Parameter(init.zeros(dim))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # x : (batch_size, feature)
        batch_size = x.shape[0]
        features = x.shape[1]

        sum_x = ops.summation(x, axes=1)
        # (batch_size, )
        mean_x = ops.divide_scalar(sum_x, features)
        # x - E[x]维度需相同, 转换为(batch_size, 1)才能广播
        tmp = ops.reshape(mean_x, (-1,1))
        broadcast_mean = ops.broadcast_to(tmp, x.shape)

        nominator = x - broadcast_mean

        sub_sqruare = nominator ** 2
        sum_sqruare = ops.summation(sub_sqruare, axes=1)
        var_x = ops.divide_scalar(sum_sqruare, features)
        # 广播var_x
        broadcast_var = ops.broadcast_to(ops.reshape(var_x,(-1,1)), x.shape)

        denominator = ops.power_scalar(broadcast_var + self.eps, 0.5)

        # weight,bias (features, ), 转换为(1,feature)才能广播
        broadcast_weight = ops.broadcast_to(ops.reshape(self.weight, (1,-1)), x.shape)
        broadcast_bias = ops.broadcast_to(ops.reshape(self.bias, (1,-1)), x.shape)

        out = broadcast_weight * nominator / denominator + broadcast_bias
        return out
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION



