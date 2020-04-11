![Logo](docs\Images\logo.png)

# Gingerbread AI

C# libraries written written to model different neural networks, and deep learning algorithms that train them.

#### What's supported?

The following neural networks are supported:

| Network                       | Supports                       |
| ----------------------------- | ------------------------------ |
| Dense Neural Networks         | Deep networks                  |
| Convolutional Neural Networks | 1D + filtering, 2D + filtering |

The following [activation functions](https://en.wikipedia.org/wiki/Activation_function) are supported:

| Function | Links                                                        |
| -------- | ------------------------------------------------------------ |
| Linear   | https://en.wikipedia.org/wiki/Identity_function              |
| RELU     | https://en.wikipedia.org/wiki/Rectifier_(neural_networks)    |
| Sigmoid  | https://en.wikipedia.org/wiki/Logistic_function              |
| Tanh     | https://en.wikipedia.org/wiki/Hyperbolic_functions#Hyperbolic_tangent |

There is an implementation of [Backpropagation](https://en.wikipedia.org/wiki/Backpropagation) (with momentum), where the following error functions are supported:

| Function           | Links                                            |
| ------------------ | ------------------------------------------------ |
| Mean Squared Error | https://en.wikipedia.org/wiki/Mean_squared_error |
| Cross Entropy      | https://en.wikipedia.org/wiki/Cross_entropy      |

## Getting Started

Open the [GingerbreadAI.NeuralNetwork.Test](NeuralNetwork/GingerbreadAI.NeuralNetwork.Test.sln) solution file.

### Prerequisites

Visual Studio 2019 (with dotnet core 3.1).

### Unit tests

All unit tests should be runnable from the [GingerbreadAI.NeuralNetwork.Test solution](src).

### Long running tests

Longer running tests that produce a report need to be run in debug mode, and can be found in the [GinbergreadAI.NeuralNetwork.Test project](src\Test\GingerbreadAI.NeuralNetwork.Test) (in the [GingerbreadAI.NeuralNetwork.Test solution)](src).

## Authors

See the list of [contributors](https://github.com/benchiverton/GingerbreadAI/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
