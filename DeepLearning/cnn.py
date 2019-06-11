from layers import *


class Linear():
    def __init__(self, in_feature, out_feature):
        self.in_feature = in_feature
        self.out_feature = out_feature

        self.W = np.random.randn(out_feature, in_feature)
        self.b = np.zeros(out_feature)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.x = x
        self.out = x.dot(self.W.T) + self.b
        return self.out

    def backward(self, delta):
        self.db = delta
        self.dW = np.dot(self.x.T, delta)
        dx = np.dot(delta, self.W.T)
        return dx


class Conv1D():
    """
    Conv(x, w) + b = z
    """

    def __init__(self, in_channel, out_channel, kernel_size, stride):

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        self.W = np.random.randn(out_channel, in_channel, kernel_size)
        self.b = np.zeros(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

        self.x = None
        self.batch = None
        self.width = None
        self.out_width = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        :param x: (batch, in_channel, width)
        :return z: (batch, out_channel/filter_num, out_width)
        """
        self.x = x
        self.batch, __, self.width = x.shape
        assert __ == self.in_channel, 'Expected the inputs to have {} channels'.format(self.in_channel)
        self.out_width = int((self.width - self.kernel_size) / self.stride) + 1
        z = np.zeros((self.batch, self.out_channel, self.out_width))
        for batch in range(self.batch):
            for j in range(self.out_channel):
                for t in range(self.out_width):
                    z[batch, j, t] = np.sum(
                        self.W[j] * x[batch, :, t * self.stride: t * self.stride + self.kernel_size],
                        keepdims=False) \
                                     + self.b[j]
        return z

    def backward(self, delta):
        """
        :param delta: dD/dz, (batch, out_channel/filter_num, out_width)
        :return: dD/dx, (batch, in_channel, width)
        """
        self.db = np.sum(np.sum(delta, 2), 0)
        dx = np.zeros(self.x.shape)
        for batch in range(self.batch):
            for j in range(self.out_channel):
                for t in range(self.out_width):
                    self.dW[j, :, :] += delta[batch, j, t] * \
                                        self.x[batch, :, t * self.stride:t * self.stride + self.kernel_size]
                    dx[batch, :, t * self.stride:t * self.stride + self.kernel_size]\
                        += delta[batch, j, t] * self.W[j, :, :]
        return dx


class Flatten():
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        :param x: (batch size, in channel, in width)
        :return y: (batch size, in channel*in width)
        """
        self.batch, self.in_channel, self.in_width = x.shape
        return x.reshape((self.batch, self.in_channel * self.in_width))

    def backward(self, delta):
        raise delta.reshape((self.batch, self.in_channel, self.in_width))


class ReLU():
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.dy = (x >= 0).astype(x.dtype)
        return x * self.dy

    def backward(self, delta):
        return self.dy * delta


class CNN_B():
    def __init__(self):
        # Conv1D (in_channel, out_channel, kernel_size/kernel_width, stride)
        self.layers = [
            # x(1, 24, 128)
            Conv1D(24, 8, 8, 4), ReLU(),
            # out1(1, 8, 31)
            Conv1D(8, 16, 1, 1), ReLU(),
            # out2(1, 16, 31)
            Conv1D(16, 4, 1, 1),
            # out3(1, 4, 31)
            Flatten()]

    def __call__(self, x):
        return self.forward(x)

    def init_weights(self, weights):
        # weight1 (out_channel=8, in_channel=24, kernel_size=8)
        for kernel in range(self.layers[0].out_channel):
            self.layers[0].W[kernel, :, :] = weights[0][:, kernel].reshape((8, 24)).T
        # weight2 (out_channel=16, in_channel=8, kernel_size=1)
        for kernel in range(self.layers[2].out_channel):
            self.layers[2].W[kernel, :, :] = weights[1][:, kernel].reshape((1, 8)).T
        # weight3 (out_channel=4, in_channel=16, kernel_size=1)
        for kernel in range(self.layers[4].out_channel):
            self.layers[4].W[kernel, :, :] = weights[2][:, kernel].reshape((1, 16)).T

    def forward(self, x):
        # You do not need to modify this method
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, delta):
        # You do not need to modify this method
        for layer in self.layers[::-1]:
            delta = layer.backward(delta)
        return delta


class CNN_C():
    def __init__(self):
        # Conv1D (in_channel, out_channel, kernel_size/kernel_width, stride)
        self.layers = [
            # x(1, 24, 128)
            Conv1D(24, 2, 2, 2), ReLU(),
            # out1(1, 2, 64)
            Conv1D(2, 8, 2, 2), ReLU(),
            # out2(1, 8, 32)
            Conv1D(8, 4, 2, 1),
            # out3(1, 4, 31)
            Flatten()]

    def __call__(self, x):
        return self.forward(x)

    def init_weights(self, weights):
        # weight1 (out_channel=2, in_channel=24, kernel_size=2)
        for kernel in range(self.layers[0].out_channel):
            self.layers[0].W[kernel, :, :] = weights[0][0:48, kernel].reshape(2, 24).T

        # weight2 (out_channel=8, in_channel=2, kernel_size=2)
        for kernel in range(self.layers[2].out_channel):
            self.layers[2].W[kernel, :, :] = weights[1][0:4, kernel].T.reshape(2, 2).T

        # weight3 (out_channel=4, in_channel=8, kernel_size=2)
        for kernel in range(self.layers[4].out_channel):
            self.layers[4].W[kernel, :, :] = weights[2][0:16, kernel].reshape(2, 8).T

    def forward(self, x):
        # You do not need to modify this method
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, delta):
        # You do not need to modify this method
        for layer in self.layers[::-1]:
            delta = layer.backward(delta)
        return delta
