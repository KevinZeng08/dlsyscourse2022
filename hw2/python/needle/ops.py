"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will numpy as the array_api
# to backup our computations, this line will change in later homeworks
import numpy as array_api


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple(*[out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x = node.inputs[0]
        return (out_grad * (self.scalar * x ** (self.scalar-1)), )
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.true_divide(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        # df_dlhs = 1 / rhs df_drhs = -lhs / (rhs ** 2)
        return (out_grad / rhs, out_grad * (-lhs) / (rhs ** 2))
        ### END YOUR SOLUTION


def divide(a, b):
    # 创建算子实例对象（EWiseDiv），并调用__call__方法，参数为(a,b)
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.true_divide(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x = node.inputs[0]
        # df_dx = 1 / self.scalar
        return (out_grad / self.scalar)
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
          return array_api.swapaxes(a, -1, -2)
        else:
          return array_api.swapaxes(a, self.axes[0], self.axes[1])
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # transpose不涉及计算，只需要将输入元素的梯度位置相应变换即可
        # Tensor.transpose定义在Tensor类中
        if self.axes is None:
          return out_grad.transpose(axes=(-1,-2))
        else:
          return out_grad.transpose(axes=(self.axes[0], self.axes[1]))
        ### END YOUR SOLUTION

def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # 逆向reshape到inputs的维度
        x = node.inputs[0]
        return out_grad.reshape(x.shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x = node.inputs[0]
        broadcast_dims = []
        in_dim = len(x.shape)
        out_dim = len(out_grad.shape)
        # 找出被广播作用的维度，然后对该维度求和（降维）
        if in_dim == out_dim:
          for i in range(in_dim):
            if x.shape[i] != out_grad.shape[i]:
              broadcast_dims.append(i)
        
        else:
          # 广播是将两个矩阵维度右对齐，因此用负索引从后向前判断
          for i in range(-1, -in_dim-1, -1):
            if x.shape[i] != out_grad.shape[i]:
              broadcast_dims.append(i)
          for i in range(-in_dim-1, -out_dim-1, -1):
            broadcast_dims.append(i)
        
        return out_grad.sum(axes=tuple(broadcast_dims)).reshape(x.shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x = node.inputs[0]

        in_dim = x.numpy().ndim
        out_dim = out_grad.numpy().ndim
        in_shape = x.numpy().shape

        if self.axes is not None:
          new_shape = list(in_shape)
          if isinstance(self.axes,int):
            new_shape[self.axes]=1
          else:
            for a in self.axes:
              new_shape[a] = 1 # 在该维度进行广播
        else:
          # reshape参数需为int类型
          new_shape = array_api.ones(len(in_shape), dtype=int)

        return out_grad.reshape(new_shape).broadcast_to(x.shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a,b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # out_grad = lhs @ rhs
        # (m*n) (n*k)
        lhs, rhs = node.inputs
        # (m*n) = (m*k) @ (k*n)
        lhs_grade = out_grad @ array_api.transpose(rhs)
        # (n*k) = (n*m) @ (m*k)
        rhs_grade = array_api.transpose(lhs) @ out_grad

        # grade可能进行了broadcast，通过sum降低维度
        if rhs_grade.shape != rhs.shape:
          # 从0开始，这是由于广播对矩阵维度进行右对齐
          rhs_grade = rhs_grade.sum(axes=tuple(range(len(rhs_grade.shape)-len(rhs.shape))))
        
        if lhs_grade.shape != lhs.shape:
          lhs_grade = lhs_grade.sum(axes=tuple(range(len(lhs_grade.shape)-len(lhs.shape))))
        return lhs_grade, rhs_grade
        ### END YOUR SOLUTION

def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.negative(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (-out_grad, )
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x = node.inputs[0]
        return (out_grad / x, )
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x = node.inputs[0]
        return (out_grad * exp(x), )
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(0,a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION        
        x = node.inputs[0]
        return out_grad * (x.realize_cached_data() > 0)
        ### END YOUR SOLUTION

def relu(a):
    return ReLU()(a)



class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        # keepdims=True, 让axes所在维度为1,可以广播
        max_z = array_api.max(Z,self.axes, keepdims=True)
        shifted_z = Z - max_z
        exp_shifted_z = array_api.exp(shifted_z)
        sum_exp = array_api.sum(exp_shifted_z,self.axes)
        log_sum_exp = array_api.log(sum_exp) + max_z.squeeze()
        return log_sum_exp
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0].cached_data
        max_z = array_api.max(Z,self.axes,keepdims=True)
        shifted_z = Z - max_z
        exp_shifted_z = array_api.exp(shifted_z)
        sum_exp = array_api.sum(exp_shifted_z, self.axes)

        log_grade = out_grad.cached_data / sum_exp

        input_shape = Z.shape
        broadcast_shape = list(input_shape)
        if self.axes is not None:
          if isinstance(self.axes, int):
            broadcast_shape[self.axes] = 1
          else:
            for i in self.axes:
              broadcast_shape[i] = 1
          log_grade = array_api.reshape(log_grade, tuple(broadcast_shape))

        exp_grade = log_grade * exp_shifted_z
        return Tensor(exp_grade) 
        ### END YOUR SOLUTION

def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

