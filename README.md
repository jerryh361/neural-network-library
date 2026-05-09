# Neural Network Library
 
A from-scratch implementation of a feedforward and convolutional neural network in **pure NumPy**, with no machine learning libraries. Backpropagation, gradient descent, and convolution are all derived and implemented by hand. Trained on MNIST, achieves **98.8%** test accuracy. 
 
> This project builds on base-model MNIST classifiers in 2 main ways:
>
> 1. **No autograd:** Every gradient, including through softmax + cross-entropy, dropout masks, and the convolution layer, is derived analytically and written out explicitly in code.
> 2. **im2col convolution:** The conv layer's forward and backward pass are vectorised via `im2col` / `col2im`, turning the four nested loops of a naive convolution into a single matrix multiplication. This is the same trick Caffe and early cuDNN used.
<img width="1188" height="1239" alt="image" src="https://github.com/user-attachments/assets/3429e4f8-c3c9-4064-a8e4-74ca98889143" />

## Project structure
 
```
neural-network-library/
├── src/
│   ├── convolutional_neural_net.py          # MLP / CNN, Stochastic Gradient Descent, save/load
│   ├── simple_neural_net.py #legacy MLP implementation
│   ├── mnist_loader.py     # MNIST loader (adapted from Nielsen, MIT)
│   └── server.py           # visualiser
├── data/                   # MNIST data
├── requirements.txt
├── licences.txt
└── README.md
```

## Results
 
| Model | Architecture | Test accuracy |
|---|---|---|
| CNN | `28×28 → Conv(20×5×5) → FC(100) → FC(10)` | **98.8%** |

 
Trained with mini-batch SGD, cross-entropy loss, ReLU activations, softmax output, L2 regularisation, and dropout on the fully-connected layer.

## Why I built this
 
I wanted to understand neural networks at the level where I could re-derive them, not just mindlessly call `model.fit()`. Implementing backprop without autograd forces you to figure out the chain rule through increasingly complex layers.  
I also learnt a lot by working out im2col convolution. Rather than using 4 nested loops, the flattening of tensors and use of matrix multiplication taught me about the various ways data manipulation can be used to optimise computation speeds.
 
## Quickstart
 
```bash
git clone https://github.com/jerryh361/neural-network-library
cd neural-network-library
pip install -r requirements.txt
```
Train a model:
 
```python
from src.network import Network
from src.mnist_loader import load_data_wrapper
 
training_data, validation_data, test_data = load_data_wrapper()
 
net = Network([
    ConvolutionalLayer(
        input_shape=(1, 28, 28),
        filter_shape=(20, 1, 5, 5),
        poolsize=(2, 2)
    ),
    FullyConnectedLayer(n_in=20*12*12, n_out=100, p_dropout=0.5),
    OutputLayer(n_in=100, n_out=10)
])
net.SGD(
    training_data,
    epochs=30,
    mini_batch_size=10,
    eta=0.1,
    lmbda=5.0,
    evaluation_data=test_data,
)
net.save("model.json")
```
 
Launch the visualiser:
 
```bash
python server.py
```

## Design decisions
- **Conv + pooling combined** As these two are often used together, it made more sense to bundle them rather than separating. This bundling also leads to cleaner code in designing network architecture.
- **L2 over L1 regularisatin** L2 is closed-form differentiable and can be implemented into the weight update quite easily. L1 needs subgradients and gave no measurable accuracy improvement on MNIST.
- **Dropout only on fully conected layers** Dropout on convolutional feature maps is counterintuitive, as contiguous activations are correlated. This is opposed to linear connected layers, which can be thought of as independent units.

## Acknowledgments
The MNIST data loader (`mnist_loader.py`) is based on the program from Michael Nielsen's
book *Neural Networks and Deep Learning*, modified to support Python 3, and is used under the MIT
License. Copyright (c) 2012–2022 Michael Nielsen.
- Book: http://neuralnetworksanddeeplearning.com
- Source repository: https://github.com/mnielsen/neural-networks-and-deep-learning
