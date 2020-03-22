using System;
using Model.NeuralNetwork;
using Model.NeuralNetwork.ActivationFunctions;
using Model.NeuralNetwork.Initialisers;
using Model.NeuralNetwork.Models;
using Xunit;

namespace DeepLearning.Backpropagation.Test
{
    public class BackpropagationShould
    {
        [Fact]
        public void TrainBasicNetworksSortofWell()
        {
            var input = new Layer(5, new Layer[0], ActivationFunctionType.RELU, InitialisationFunctionType.GlorotUniform);
            var h1 = new Layer(10, new Layer[] { input }, ActivationFunctionType.RELU, InitialisationFunctionType.GlorotUniform);
            var output = new Layer(5, new Layer[] { h1 }, ActivationFunctionType.Sigmoid, InitialisationFunctionType.HeEtAl);

            output.Initialise(new Random());

            var inputs = new double[] { 0.1, 0.2, 0.3, 0.4, 0.5 };
            var targetOutputs = new double[] { 1, 0, 1, 0, 1 };
            var learningRate = 0.25;
            for (var i = 0; i < 1000; i++)
            {
                output.Backpropagate(inputs, targetOutputs, learningRate);
            }

            var outputResults = output.GetResults(inputs);

            Assert.True(outputResults[0] > 0.95);
            Assert.True(outputResults[2] > 0.95);
            Assert.True(outputResults[4] > 0.95);

            Assert.True(outputResults[1] < 0.05);
            Assert.True(outputResults[3] < 0.05);
        }

        [Fact]
        public void TrainComplexNetworksSortofWell()
        {
            var input = new Layer(5, new Layer[0], ActivationFunctionType.RELU, InitialisationFunctionType.GlorotUniform);
            var h1 = new Layer(10, new Layer[] { input }, ActivationFunctionType.RELU, InitialisationFunctionType.GlorotUniform);
            var h2 = new Layer(10, new Layer[] { input }, ActivationFunctionType.RELU, InitialisationFunctionType.GlorotUniform);
            var h3 = new Layer(10, new Layer[] { h1, h2 }, ActivationFunctionType.RELU, InitialisationFunctionType.GlorotUniform);
            var output = new Layer(5, new Layer[] { h3 }, ActivationFunctionType.Sigmoid, InitialisationFunctionType.HeEtAl);

            output.Initialise(new Random());

            var inputs = new double[] { 0.1, 0.2, 0.3, 0.4, 0.5 };
            var targetOutputs = new double[] { 1, 0, 1, 0, 1 };
            var learningRate = 0.25;
            for (var i = 0; i < 1000; i++)
            {
                output.Backpropagate(inputs, targetOutputs, learningRate);
            }

            var outputResults = output.GetResults(inputs);

            Assert.True(outputResults[0] > 0.95);
            Assert.True(outputResults[2] > 0.95);
            Assert.True(outputResults[4] > 0.95);

            Assert.True(outputResults[1] < 0.05);
            Assert.True(outputResults[3] < 0.05);
        }
    }
}