# Gingerbread AI

C# libraries written written to model neural networks, and deep learning algorithms to train them.

## [NeuralNetwork](NeuralNetwork)

Contains C# libraries that model the following neural networks:

* Fully Connected (Dense) Neural Networks
* Convolutional Neural Networks (1D w/ filtering, 2D w/ filtering)

The following activation functions are supported:
* Linear
* RELU
* Sigmoid
* Tanh

There is an implementation of Baskpropagation (w/ momentum), where the following error/loss functions are supported:
* MSE
* Cross Entropy

#### [Documentation](Documentation)

Various documentation that has been written explaining some of the decisions taken when writing the GingerbreadAI library.

## Getting Started

Open the [GingerbreadAI.NeuralNetwork.Test](NeuralNetwork/GingerbreadAI.NeuralNetwork.Test.sln) solution file.

### Prerequisites

Visual Studio 2019 (with dotnet core 3.1)

## Unit tests

All unit tests should be runnable from the [GingerbreadAI.NeuralNetwork.Test](NeuralNetwork/GingerbreadAI.NeuralNetwork.Test.sln) solution.

### Long running tests

Longer running tests that produce a report can only be run in debug, and can be found in the GinbergreadAI.NeuralNetwork.Tests project, which is in the [GingerbreadAI.NeuralNetwork.Test](NeuralNetwork/GingerbreadAI.NeuralNetwork.Test.sln) solution.

## Authors

See the list of [contributors](https://github.com/benchiverton/GingerbreadAI/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
