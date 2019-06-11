import numpy as np
import itertools


class Sigmoid:
    """docstring for Sigmoid"""

    def __init__(self):
        pass

    def forward(self, x):
        self.res = 1 / (1 + np.exp(-x))
        return self.res

    def backward(self):
        return self.res * (1 - self.res)

    def __call__(self, x):
        return self.forward(x)


class Tanh:
    def __init__(self):
        pass

    def forward(self, x):
        self.res = np.tanh(x)
        return self.res

    def backward(self):
        return 1 - (self.res ** 2)

    def __call__(self, x):
        return self.forward(x)


class GRU_Cell:
    """docstring for GRU_Cell"""

    def __init__(self, in_dim, hidden_dim):
        self.d = in_dim
        self.h = hidden_dim
        h = self.h
        d = self.d

        self.Wzh = np.random.randn(h, h)
        self.Wrh = np.random.randn(h, h)
        self.Wh = np.random.randn(h, h)

        self.Wzx = np.random.randn(h, d)
        self.Wrx = np.random.randn(h, d)
        self.Wx = np.random.randn(h, d)

        self.dWzh = np.zeros((h, h))
        self.dWrh = np.zeros((h, h))
        self.dWh = np.zeros((h, h))

        self.dWzx = np.zeros((h, d))
        self.dWrx = np.zeros((h, d))
        self.dWx = np.zeros((h, d))

        self.z_act = Sigmoid()
        self.r_act = Sigmoid()
        self.h_act = Tanh()

    def forward(self, x, h):
        # input:
        # 	- x: shape(input dim),  observation at current time-step
        # 	- h: shape(hidden dim), hidden-state at previous time-step
        #
        # output:
        # 	- h_t: hidden state at current time-step
        self.h_p = h
        self.x_t = x
        self.z = self.z_act(self.Wzh.dot(self.h_p) + self.Wzx.dot(x))
        self.r = self.r_act(self.Wrh.dot(self.h_p) + self.Wrx.dot(x))
        self.h_tilde = self.h_act(self.Wh.dot(self.r * self.h_p) + self.Wx.dot(x))
        self.h_t = (1 - self.z) * self.h_p + self.z * self.h_tilde
        return self.h_t

    def backward(self, delta):
        # input:
        # 	- delta: 	shape(hidden dim), summation of derivative wrt loss from next layer at
        # 			same time-step and derivative wrt loss from same layer at
        # 			next time-step
        #
        # output:
        # 	- dx: 	Derivative of loss wrt the input x
        # 	- dh: 	Derivative of loss wrt the input hidden h
        dz = delta * (self.h_tilde - self.h_p)
        self.dWzx += np.outer(dz * self.z_act.backward(), self.x_t)
        self.dWzh += np.outer(dz * self.z_act.backward(), self.h_p)

        dh_tilde = delta * self.z
        self.dWx += np.outer(dh_tilde * self.h_act.backward(), self.x_t)
        self.dWh += np.outer(dh_tilde * self.h_act.backward(), self.r * self.h_p)

        dr = (dh_tilde * self.h_act.backward()).dot(self.Wh) * self.h_p
        self.dWrx += np.outer(dr * self.r_act.backward(), self.x_t)
        self.dWrh += np.outer(dr * self.r_act.backward(), self.h_p)

        dx = (dz * self.z_act.backward()).dot(self.Wzx) \
             + (dh_tilde * self.h_act.backward()).dot(self.Wx) \
             + (dr * self.r_act.backward()).dot(self.Wrx)
        dh = (dz * self.z_act.backward()).dot(self.Wzh) \
             + (dr * self.r_act.backward()).dot(self.Wrh) \
             + (dh_tilde * self.h_act.backward()).dot(self.Wh) * self.r\
             + (delta * (1 - self.z))

        return dx, dh


if __name__ == '__main__':
    # test()
    self = GRU_Cell(4, 8)
    h = np.random.randn(8)
    x = np.random.randn(4)
    delta = np.random.randn(8)
    h = self.forward(x, h)
    dx, dh = self.backward(delta)
    print(dx.shape, dh.shape)
