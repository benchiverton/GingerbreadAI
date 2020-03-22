using System;
using System.Collections.Generic;
using System.Linq;
using DeepLearning.Backpropagation.Extensions;
using Model.ConvolutionalNeuralNetwork.Extensions;
using Model.ConvolutionalNeuralNetwork.Models;
using Model.NeuralNetwork;
using Model.NeuralNetwork.ActivationFunctions;
using Model.NeuralNetwork.Initialisers;
using Model.NeuralNetwork.Models;
using Xunit;
using Xunit.Abstractions;

namespace DeepLearning.Backpropagation.CNN.Test
{
    public class BackPropagatorWith1DNetworkShould
    {
        private readonly ITestOutputHelper _testOutputHelper;

        public BackPropagatorWith1DNetworkShould(ITestOutputHelper testOutputHelper)
        {
            _testOutputHelper = testOutputHelper;
        }

        [Fact]
        public void TrainFilterToFeatureSortOfWell()
        {
            var inputLayer = new Layer1D(3, new Layer[0], ActivationFunctionType.RELU, InitialisationFunctionType.None);
            var filter = new Filter1D(new[] { inputLayer }, 3, ActivationFunctionType.RELU, InitialisationFunctionType.HeUniform);
            var output = new Layer(1, new Layer[] { filter }, ActivationFunctionType.Sigmoid, InitialisationFunctionType.HeEtAl);
            var momentum = output.GenerateMomentum();
            output.Initialise(new Random());
            var inputMatch = new double[] { 1, 0, 1 };
            var inputNoMatch = new double[] { 0, 1, 0 };

            for (var i = 0; i < 10000; i++)
            {
                output.Backpropagate(inputMatch, new double[] { 1 }, 0.1, momentum, 0.9);
                output.Backpropagate(inputNoMatch, new double[] { 0 }, 0.1, momentum, 0.9);
            }

            // filter weights should look as follows:
            // +  -  +      -  +  -  (Depending on output weight being +ve/-ve)
            _testOutputHelper.WriteLine($"filter: {string.Join(",", filter.Nodes[0].Weights.Values.Select(v => v.Value.ToString("0.00")))}");

            var outputMultiplier = output.Nodes[0].Weights[filter.Nodes[0]].Value > 0 ? 1 : -1;
            Assert.True(outputMultiplier * filter.Nodes[0].Weights[inputLayer.Nodes[0]].Value > 0);
            Assert.True(outputMultiplier * filter.Nodes[0].Weights[inputLayer.Nodes[1]].Value < 0);
            Assert.True(outputMultiplier * filter.Nodes[0].Weights[inputLayer.Nodes[2]].Value > 0);
        }

        [Fact]
        public void TrainFilterToRBGFeatureSortOfWell()
        {
            var r = new Layer1D(4, new Layer[0], ActivationFunctionType.RELU, InitialisationFunctionType.None);
            var g = new Layer1D(4, new Layer[0], ActivationFunctionType.RELU, InitialisationFunctionType.None);
            var b = new Layer1D(4, new Layer[0], ActivationFunctionType.RELU, InitialisationFunctionType.None);
            var filter1 = new Filter1D(new[] { r, g, b }, 2, ActivationFunctionType.RELU, InitialisationFunctionType.HeUniform);
            var filter2 = new Filter1D(new[] { r, g, b }, 2, ActivationFunctionType.RELU, InitialisationFunctionType.HeUniform);
            var filter3 = new Filter1D(new[] { r, g, b }, 2, ActivationFunctionType.RELU, InitialisationFunctionType.HeUniform);
            var output = new Layer(3, new Layer[] { filter1, filter2, filter3 }, ActivationFunctionType.Sigmoid, InitialisationFunctionType.HeEtAl);
            output.Initialise(new Random());
            Dictionary<Layer, double[]> ResolveInputs(bool isRed, bool isGreen, bool isBlue)
            {
                var rInput = isRed
                    ? new double[] { 1, 1, 1, 1 }
                    : new double[] { 0, 0, 0, 0 };
                var gInput = isGreen
                    ? new double[] { 1, 1, 1, 1 }
                    : new double[] { 0, 0, 0, 0 };
                var bInput = isBlue
                    ? new double[] { 1, 1, 1, 1 }
                    : new double[] { 0, 0, 0, 0 };
                return new Dictionary<Layer, double[]>
                {
                    [r] = rInput,
                    [g] = gInput,
                    [b] = bInput
                };
            }

            for (var i = 0; i < 10000; i++)
            {
                var isRed = i % 3 == 0;
                var isGreen = i % 3 == 1;
                var isBlue = i % 3 == 2;
                var inputs = ResolveInputs(isRed, isGreen, isBlue);
                var targetOutputs = new[] { isRed ? 1d : 0d, isGreen ? 1d : 0d, isBlue ? 1d : 0d };

                output.Backpropagate(inputs, targetOutputs, 0.5);
            }

            // each filter should pick up r/b/g differently
            _testOutputHelper.WriteLine($"filter 1: {string.Join(",", filter1.Nodes[0].Weights.Values.Select(v => v.Value.ToString("0.00")))}");
            _testOutputHelper.WriteLine($"filter 2: {string.Join(",", filter2.Nodes[0].Weights.Values.Select(v => v.Value.ToString("0.00")))}");
            _testOutputHelper.WriteLine($"filter 3: {string.Join(",", filter3.Nodes[0].Weights.Values.Select(v => v.Value.ToString("0.00")))}");
            var redInput = ResolveInputs(true, false, false);
            output.CalculateOutputs(redInput);
            Assert.True(output.Nodes[0].Output > 0.95);
            Assert.True(output.Nodes[1].Output < 0.05);
            Assert.True(output.Nodes[2].Output < 0.05);
            var greenInput = ResolveInputs(false, true, false);
            output.CalculateOutputs(greenInput);
            Assert.True(output.Nodes[0].Output < 0.05);
            Assert.True(output.Nodes[1].Output > 0.95);
            Assert.True(output.Nodes[2].Output < 0.05);
            var blueInput = ResolveInputs(false, false, true);
            output.CalculateOutputs(blueInput);
            Assert.True(output.Nodes[0].Output < 0.05);
            Assert.True(output.Nodes[1].Output < 0.05);
            Assert.True(output.Nodes[2].Output > 0.95);
        }

        [Fact]
        public void TrainNetworkWithFilterSortOfWellRgb()
        {
            var r = new Layer1D(4, new Layer[0], ActivationFunctionType.RELU, InitialisationFunctionType.None);
            var g = new Layer1D(4, new Layer[0], ActivationFunctionType.RELU, InitialisationFunctionType.None);
            var b = new Layer1D(4, new Layer[0], ActivationFunctionType.RELU, InitialisationFunctionType.None);
            var filters = new[]
            {
                new Filter1D(new[] {r, g, b}, 2, ActivationFunctionType.RELU, InitialisationFunctionType.HeUniform),
                new Filter1D(new[] {r, g, b}, 2, ActivationFunctionType.RELU, InitialisationFunctionType.HeUniform),
                new Filter1D(new[] {r, g, b}, 2, ActivationFunctionType.RELU, InitialisationFunctionType.HeUniform)
            };
            filters.AddPooling(2);
            var output = new Layer(3, filters, ActivationFunctionType.Sigmoid, InitialisationFunctionType.HeEtAl);
            var momentum = output.GenerateMomentum();
            output.Initialise(new Random());
            Dictionary<Layer, double[]> ResolveInputs(bool isRed, bool isGreen, bool isBlue)
            {
                var rInput = isRed
                    ? new double[] { 1, 1, 1, 1 }
                    : new double[] { 0, 0, 0, 0 };
                var gInput = isGreen
                    ? new double[] { 1, 1, 1, 1 }
                    : new double[] { 0, 0, 0, 0 };
                var bInput = isBlue
                    ? new double[] { 1, 1, 1, 1 }
                    : new double[] { 0, 0, 0, 0 };
                return new Dictionary<Layer, double[]>
                {
                    [r] = rInput,
                    [g] = gInput,
                    [b] = bInput
                };
            }

            for (var i = 0; i < 10000; i++)
            {
                var isRed = i % 3 == 0;
                var isGreen = i % 3 == 1;
                var isBlue = i % 3 == 2;
                var inputs = ResolveInputs(isRed, isGreen, isBlue);
                var targetOutputs = new[] { isRed ? 1d : 0d, isGreen ? 1d : 0d, isBlue ? 1d : 0d };

                output.Backpropagate(inputs, targetOutputs, 0.1, momentum, 0.9);
            }

            var redInput = ResolveInputs(true, false, false);
            output.CalculateOutputs(redInput);
            Assert.True(output.Nodes[0].Output > 0.95);
            Assert.True(output.Nodes[1].Output < 0.05);
            Assert.True(output.Nodes[2].Output < 0.05);
            var greenInput = ResolveInputs(false, true, false);
            output.CalculateOutputs(greenInput);
            Assert.True(output.Nodes[0].Output < 0.05);
            Assert.True(output.Nodes[1].Output > 0.95);
            Assert.True(output.Nodes[2].Output < 0.05);
            var blueInput = ResolveInputs(false, false, true);
            output.CalculateOutputs(blueInput);
            Assert.True(output.Nodes[0].Output < 0.05);
            Assert.True(output.Nodes[1].Output < 0.05);
            Assert.True(output.Nodes[2].Output > 0.95);
        }
    }
}