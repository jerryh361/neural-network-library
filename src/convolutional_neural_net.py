
import json
import numpy as np


####################activation functions####################
def relu(z):
    return np.maximum(0, z)

def relu_prime(z):
    return (z > 0).astype(float)


def softmax(z):
    # subtract max for numerical stability; z shape (n_out, N)
    e = np.exp(z - z.max(axis=0, keepdims=True))
    return e / e.sum(axis=0, keepdims=True)


 ###################cost functions##########################
class CrossEntropyCost(object):
    @staticmethod
    def c(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(a, y):
        return a-y

class CategoricalCrossEntropyCost(object):
    @staticmethod
    def c(a, y):
        return -np.sum(np.nan_to_num(y * np.log(a)))

    @staticmethod
    def delta(a, y):
        # combined derivative of softmax + categorical cross-entropy simplifies to a - y
        return a - y
    
###################network object#########################
class Network(object):
    def __init__(self, layers, cost=CategoricalCrossEntropyCost):
        self.layers = layers
        self.cost = cost

    def set_training(self, mode):
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = mode

    def feedforward(self, inpt, mini_batch_size):
        for layer in self.layers:
            layer.set_input(inpt, mini_batch_size)
            inpt = layer.output
        return inpt

    def backprop(self, x, y, mini_batch_size):
        # forward pass (training mode for dropout)
        self.set_training(True)
        self.feedforward(x, mini_batch_size)
        for i, layer in enumerate(self.layers):
            if np.any(np.isnan(layer.output)) or np.any(np.isinf(layer.output)):
                print(f"NaN/Inf detected in layer {i} output")
                return [np.zeros_like(l.w) for l in self.layers], \
                       [np.zeros_like(l.b) for l in self.layers]
        # backward pass
        nabla_w = [None] * len(self.layers)
        nabla_b = [None] * len(self.layers)

        # output layer uses cost delta directly
        delta, nabla_w[-1], nabla_b[-1] = self.layers[-1].backprop_output(y, self.cost)

        # propagate backwards through remaining layers
        for l in range(len(self.layers)-2, -1, -1):
            # reshape delta to match layer output if transitioning conv->fc
            delta, nabla_w[l], nabla_b[l] = self.layers[l].backprop(delta)

        return nabla_w, nabla_b

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        x = np.array([pair[0] for pair in mini_batch])
        y = np.array([pair[1] for pair in mini_batch]).squeeze(axis=-1).T

        nabla_w, nabla_b = self.backprop(x, y, len(mini_batch))

        # clip by global norm , prevent gradient blowup.
        all_grads = [g for nw in nabla_w for g in [nw]] + [g for nb in nabla_b for g in [nb]]
        global_norm = np.sqrt(sum(np.sum(g**2) for g in all_grads))
        clip_threshold = 1.0
        if global_norm > clip_threshold:
            scale = clip_threshold / global_norm
            nabla_w = [nw * scale for nw in nabla_w]
            nabla_b = [nb * scale for nb in nabla_b]

        for layer, nw, nb in zip(self.layers, nabla_w, nabla_b):
            layer.w = (1 - eta*(lmbda/n))*layer.w - eta*nw
            layer.b = layer.b - eta*nb


    def SGD(self, training_data, epochs, mini_batch_size, eta, lmbda=0.0, evaluation_data=None):
        training_data = list(training_data)
        n = len(training_data)
        if evaluation_data is not None:
            evaluation_data = list(evaluation_data)
            n_eval = len(evaluation_data)

        for j in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, n)

            if evaluation_data is not None:
                acc = self.accuracy(evaluation_data)
                print(f"Epoch {j}: {acc} / {n_eval}")
            else:
                print(f"Epoch {j} complete")

    def save(self, filename):
        data = {
            "cost": self.cost.__name__,
            "layers": [layer.to_dict() for layer in self.layers],
        }
        with open(filename, "w") as f:
            json.dump(data, f)

    @staticmethod
    def load(filename):
        with open(filename, "r") as f:
            data = json.load(f)
        cost_cls = {
            "CategoricalCrossEntropyCost": CategoricalCrossEntropyCost,
            "CrossEntropyCost": CrossEntropyCost,
        }[data["cost"]]
        layer_cls = {
            "ConvolutionalLayer": ConvolutionalLayer,
            "FullyConnectedLayer": FullyConnectedLayer,
            "OutputLayer": OutputLayer,
        }
        layers = [layer_cls[d["type"]].from_dict(d) for d in data["layers"]]
        return Network(layers, cost=cost_cls)

    def accuracy(self, data):
        self.set_training(False)
        # run one image at a time for evaluation
        results = []
        for x, y in data:
            # x shape from mnist_loader: (784, 1) — reshape to (1, 1, 28, 28)
            inpt = x.reshape(1, 1, 28, 28)
            output = self.feedforward(inpt, mini_batch_size=1)
            results.append((np.argmax(output), y))
        return sum(int(x == y) for x, y in results)

class ConvolutionalLayer(object):
    def __init__(self, input_shape, filter_shape, poolsize):
        self.input_shape=input_shape
        self.filter_shape = filter_shape
        self.poolsize = poolsize
        #n_out= (filter_shape[0]*np.prod(filter_shape[2:])/np.prod(poolsize))
        n_in = filter_shape[1] * np.prod(filter_shape[2:])
        self.w = np.random.randn(*filter_shape)/np.sqrt(n_in)
        self.b = np.zeros(filter_shape[0])                                    #intialise shared weights + biases

    def im2col(self, inpt, filter_h, filter_w, stride=1):
        N, C, H, W = inpt.shape
        out_h = (H - filter_h) // stride + 1
        out_w = (W - filter_w) // stride + 1
        col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
        for y in range(filter_h):
            for x in range(filter_w):
                col[:, :, y, x, :, :] = inpt[
                    :, :,
                    y:y + stride*out_h:stride,
                    x:x + stride*out_w:stride
                ]
      
        return col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    
    def col2im(self, dcol, inpt_shape, filter_h, filter_w, stride=1):
        N, C, H, W = inpt_shape
        out_h = (H - filter_h) // stride + 1
        out_w = (W - filter_w) // stride + 1
        col = dcol.reshape(N, out_h, out_w, C, filter_h, filter_w)
        col = col.transpose(0, 3, 4, 5, 1, 2)  # (N, C, fh, fw, out_h, out_w)
        dinpt = np.zeros((N, C, H, W))
        for y in range(filter_h):
            for x in range(filter_w):
                dinpt[
                    :, :,
                    y:y + stride*out_h:stride,
                    x:x + stride*out_w:stride
                ] += col[:, :, y, x, :, :]
        return dinpt
    
    def set_input(self, inpt, mini_batch_size):
        self.inpt = inpt.reshape(mini_batch_size, *self.input_shape)
        n_filters, n_channels, fh, fw = self.filter_shape
        out_h = self.input_shape[1] - fh + 1
        out_w = self.input_shape[2] - fw + 1

        # im2col forward pass
        self.col = self.im2col(self.inpt, fh, fw)        # (N*out_h*out_w, C*fh*fw)
        w_col = self.w.reshape(n_filters, -1)             # (n_filters, C*fh*fw)

        # convolution by matrix multiplication
        out = (self.col @ w_col.T + self.b.T)             # (N*out_h*out_w, n_filters)
        out = out.reshape(mini_batch_size, out_h, out_w, n_filters)
        out = out.transpose(0, 3, 1, 2)                   # (N, n_filters, out_h, out_w)

        # activation
        self.z_conv = out
        out = relu(out)

        # max pooling
        self.output = self.max_pool(out)

    def max_pool(self, inpt):
        N, C, H, W = inpt.shape
        ph, pw = self.poolsize
        out_h = H // ph
        out_w = W // pw
        # reshape to expose pooling windows
        x = inpt.reshape(N, C, out_h, ph, out_w, pw)
        self.pool_input = inpt                            # save for backprop
        return x.max(axis=(3, 5))
    
    def backprop(self, delta):
        
        n_filters, n_channels, fh, fw = self.filter_shape
        N, C, H, W = self.inpt.shape
        out_h = H - fh + 1
        out_w = W - fw + 1
        ph, pw = self.poolsize
        pool_out_h = out_h // ph
        pool_out_w = out_w // pw

        # delta arrives as (n_filters * pool_out_h * pool_out_w, N) from FC layer
        # reshape to (N, n_filters, pool_out_h, pool_out_w)
        delta = delta.T.reshape(N, n_filters, pool_out_h, pool_out_w)

        # now safe to backprop through pooling
        delta = self.max_pool_backprop(delta)   # (N, n_filters, out_h, out_w)

        # backprop
        delta = delta * relu_prime(self.z_conv)

        # backprop through conv
        delta_flat = delta.transpose(0, 2, 3, 1).reshape(-1, n_filters)
        dw = (delta_flat.T @ self.col).reshape(self.filter_shape) / N
        db = delta_flat.sum(axis=0).reshape(self.b.shape) / N
        w_col = self.w.reshape(n_filters, -1)
        dcol = delta_flat @ w_col
        dinpt = self.col2im(dcol, self.inpt.shape, fh, fw)

        return dinpt, dw, db
    
    def to_dict(self):
        return {
            "type": "ConvolutionalLayer",
            "input_shape": list(self.input_shape),
            "filter_shape": list(self.filter_shape),
            "poolsize": list(self.poolsize),
            "w": self.w.tolist(),
            "b": self.b.tolist(),
        }

    @classmethod
    def from_dict(cls, d):
        layer = cls(
            input_shape=tuple(d["input_shape"]),
            filter_shape=tuple(d["filter_shape"]),
            poolsize=tuple(d["poolsize"]),
        )
        layer.w = np.array(d["w"])
        layer.b = np.array(d["b"])
        return layer

    def max_pool_backprop(self, delta):
        N, C, H, W = self.pool_input.shape
        ph, pw = self.poolsize
        out_h = H // ph
        out_w = W // pw

        # find which elements were the max in each window
        x = self.pool_input.reshape(N, C, out_h, ph, out_w, pw)
        x_max = x.max(axis=(3, 5), keepdims=True)
        mask = (x == x_max).astype(float)
        mask /= mask.sum(axis=(3, 5), keepdims=True)       # split gradient among ties

        # distribute gradient only to max elements
        delta_expanded = delta[:, :, :, np.newaxis, :, np.newaxis]
        return (mask * delta_expanded).reshape(N, C, H, W)
    
    


class FullyConnectedLayer(object):
    def __init__(self, n_in, n_out, activation_fn=relu, activation_fn_prime=relu_prime, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.activation_fn_prime = activation_fn_prime
        self.p_dropout = p_dropout
        self.training = True
        self.w = np.random.randn(n_out, n_in) / np.sqrt(n_in / 2)
        self.b = np.zeros((n_out, 1))

    def set_input(self, inpt, mini_batch_size):
        # Accept either (N, C, H, W)/(N, ...) from a conv/input layer, or
        # (n_in, N) from a previous FC layer. Detect when FC->FC.
        if inpt.ndim == 2 and inpt.shape[1] == mini_batch_size:
            self.inpt = inpt                                  # already (n_in, N)
        else:
            self.inpt = inpt.reshape(mini_batch_size, -1).T   # (n_in, N)

        self.z = np.dot(self.w, self.inpt) + self.b      # (n_out, N)
        self.output = self.activation_fn(self.z)          # (n_out, N)
        #  drouput
        if self.training and self.p_dropout > 0.0:
            self.dropout_mask = (np.random.rand(*self.output.shape) > self.p_dropout) / (1.0 - self.p_dropout)
            self.output = self.output * self.dropout_mask

    def backprop(self, delta):
        # backprop through dropout
        if self.p_dropout > 0.0:
            delta = delta * self.dropout_mask             # (n_out, N)
        rp = self.activation_fn_prime(self.z)
        delta = delta * rp                                # (n_out, N)
        dw = np.dot(delta, self.inpt.T) / delta.shape[1] # (n_out, n_in)
        db = delta.mean(axis=1, keepdims=True)            # (n_out, 1)
        dinpt = np.dot(self.w.T, delta)                   # (n_in, N)
        # reshape dinpt back to conv output shape if needed
        return dinpt, dw, db

    def to_dict(self):
        return {
            "type": "FullyConnectedLayer",
            "n_in": self.n_in,
            "n_out": self.n_out,
            "p_dropout": self.p_dropout,
            "w": self.w.tolist(),
            "b": self.b.tolist(),
        }

    @classmethod
    def from_dict(cls, d):
        layer = cls(n_in=d["n_in"], n_out=d["n_out"], p_dropout=d.get("p_dropout", 0.0))
        layer.w = np.array(d["w"])
        layer.b = np.array(d["b"])
        return layer


class OutputLayer(FullyConnectedLayer):
    def __init__(self, n_in, n_out):
        super().__init__(n_in, n_out, activation_fn=softmax)

    def backprop_output(self, y, cost=CategoricalCrossEntropyCost):
        delta = cost.delta(self.output, y)                # (n_out, N)
        dw = np.dot(delta, self.inpt.T) / delta.shape[1]
        db = delta.mean(axis=1, keepdims=True)
        dinpt = np.dot(self.w.T, delta)
        return dinpt, dw, db

    def to_dict(self):
        return {
            "type": "OutputLayer",
            "n_in": self.n_in,
            "n_out": self.n_out,
            "w": self.w.tolist(),
            "b": self.b.tolist(),
        }

    @classmethod
    def from_dict(cls, d):
        layer = cls(n_in=d["n_in"], n_out=d["n_out"])
        layer.w = np.array(d["w"])
        layer.b = np.array(d["b"])
        return layer