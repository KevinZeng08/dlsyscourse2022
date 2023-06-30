"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        # weight_decay在loss上加了一个正则项，求导后为weight_decay * param
        for p in self.params:
            grad = self.u.get(p, 0) * self.momentum + (1 - self.momentum) * (p.grad.detach() + self.weight_decay * p.detach())
            grad = ndl.Tensor(grad, device=p.device, dtype=p.dtype)
            self.u[p] = grad
            # 左边的p.data调用@data.setter修饰的data函数
            # 右边的p.data新创建一个独立于计算图的Tensor进行计算，调用@property修饰的data
            p.data =  p.data - self.lr * grad
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for p in self.params:
            grad_with_wd = p.grad.detach() + self.weight_decay * p.detach()
            new_m = (self.m.get(p, 0) * self.beta1 + (1 - self.beta1) * grad_with_wd).detach()
            new_v = (self.v.get(p, 0) * self.beta2 + (1 - self.beta2) * grad_with_wd * grad_with_wd).detach()

            self.m[p] = new_m
            self.v[p] = new_v

            m_with_bias_corr = (new_m / (1 - self.beta1 ** self.t)).detach()
            v_with_bias_corr = (new_v / (1 - self.beta2 ** self.t)).detach()

            update = (self.lr * m_with_bias_corr / (ndl.power_scalar(v_with_bias_corr, 0.5).detach() + self.eps)).detach()
            update = ndl.Tensor(update, dtype=p.dtype)
            p.data -= update.detach()
        ### END YOUR SOLUTION
