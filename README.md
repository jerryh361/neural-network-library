# Neural Network Library
This library implements a simple and a convolutional neural network (CNN) for handwritten digit recognition based on the MNIST dataset. ReLU activation function + softmax is used, as well as L2 regularisation and dropout.
I achieved 98.5% accuracy, however with further hyperparameter optimisation I suspect better results are possible.
<img width="1188" height="1239" alt="image" src="https://github.com/user-attachments/assets/3429e4f8-c3c9-4064-a8e4-74ca98889143" />
The CNN uses the im2col method for backpropagation. Essentially, instead of using an inefficient nested loop, the relevant tensor operations are "flattened" into one big matrix multiplication. By using already optimised matrix multiplication libraries, a far better time-efficiency can be achieved at the cost of more memory.
# Usage

## Visualisation
```
python server.py #open visualiser server.
```

## training and saving
If you want to test your own hyperparameters and see how they affect accuracy, you can run the following code.
```
net.SGD(training_data, epochs, mini_batch_size, eta, lambda, evaluation_data) #leave evaluation_data = None for training without evaluation.
net.save("model.json")
```  
