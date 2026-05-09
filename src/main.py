import mnist_loader
from convolutional_neural_net import Network, ConvolutionalLayer, FullyConnectedLayer, OutputLayer


training_data, validation_data, evaluation_data = mnist_loader.load_data_wrapper()

net = Network([
    ConvolutionalLayer(
        input_shape=(1, 28, 28),
        filter_shape=(20, 1, 5, 5),
        poolsize=(2, 2)
    ),
    FullyConnectedLayer(n_in=20*12*12, n_out=100, p_dropout=0.5),
    OutputLayer(n_in=100, n_out=10)
])

net.SGD(training_data, 5, 32, 0.3, 0.0, evaluation_data=evaluation_data)
net.save("model.json")
print("Saved model to model.json")