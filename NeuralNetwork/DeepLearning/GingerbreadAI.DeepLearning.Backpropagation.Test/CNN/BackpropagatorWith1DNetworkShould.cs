using System;
using System.Collections.Generic;
using System.Linq;
using GingerbreadAI.DeepLearning.Backpropagation.ErrorFunctions;
using GingerbreadAI.DeepLearning.Backpropagation.Extensions;
using GingerbreadAI.Model.ConvolutionalNeuralNetwork.Extensions;
using GingerbreadAI.Model.ConvolutionalNeuralNetwork.Models;
using GingerbreadAI.Model.NeuralNetwork.ActivationFunctions;
using GingerbreadAI.Model.NeuralNetwork.Extensions;
using GingerbreadAI.Model.NeuralNetwork.InitialisationFunctions;
using GingerbreadAI.Model.NeuralNetwork.Models;
using Xunit;
using Xunit.Abstractions;

namespace GingerbreadAI.DeepLearning.Backpropagation.Test.CNN
{
    public class BackpropagatorWith1DNetworkShould
    {
        private readonly ITestOutputHelper _testOutputHelper;

        public BackpropagatorWith1DNetworkShould(ITestOutputHelper testOutputHelper)
        {
            _testOutputHelper = testOutputHelper;
        }

        [Fact]
        public void TrainFilterToFeatureSortOfWell()
        {
            var inputLayer = new Layer1D(10, new Layer[0], ActivationFunctionType.RELU, InitialisationFunctionType.None);
            var filter1 = new Filter1D(new[] { inputLayer }, 3, ActivationFunctionType.RELU, InitialisationFunctionType.GlorotUniform);
            var filter2 = new Filter1D(new[] { inputLayer }, 3, ActivationFunctionType.RELU, InitialisationFunctionType.GlorotUniform);
            var output = new Layer(1, new Layer[] { filter1, filter2 }, ActivationFunctionType.Sigmoid, InitialisationFunctionType.HeEtAl);
            output.AddMomentumRecursively();
            output.Initialise(new Random());
            var inputMatch1 = new double[] { 1, 1, 1, 0, 0, 0, 0, 0, 0, 0 };
            var inputMatch2 = new double[] { 0, 0, 0, 1, 1, 1, 0, 0, 0, 0 };
            var inputMatch3 = new double[] { 0, 0, 0, 0, 0, 0, 0, 1, 1, 1 };
            var inputNoMatch1 = new double[] { 0, 1, 0, 1, 0, 1, 0, 1, 0, 1 };
            var inputNoMatch2 = new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
            var inputNoMatch3 = new double[] { 1, 0, 0, 1, 0, 0, 1, 0, 0, 1 };

            for (var i = 0; i < 10000; i++)
            {
                output.Backpropagate(inputMatch1, new double[] { 1 }, ErrorFunctionType.CrossEntropy, 0.01, 0.9);
                output.Backpropagate(inputMatch2, new double[] { 1 }, ErrorFunctionType.CrossEntropy, 0.01, 0.9);
                output.Backpropagate(inputMatch3, new double[] { 1 }, ErrorFunctionType.CrossEntropy, 0.01, 0.9);
                output.Backpropagate(inputNoMatch1, new double[] { 0 }, ErrorFunctionType.CrossEntropy, 0.01, 0.9);
                output.Backpropagate(inputNoMatch2, new double[] { 0 }, ErrorFunctionType.CrossEntropy, 0.01, 0.9);
                output.Backpropagate(inputNoMatch3, new double[] { 0 }, ErrorFunctionType.CrossEntropy, 0.01, 0.9);
            }

            output.CalculateOutputs(inputMatch1);
            Assert.True(output.Nodes[0].Output > 0.95);
            output.CalculateOutputs(inputMatch2);
            Assert.True(output.Nodes[0].Output > 0.95);
            output.CalculateOutputs(inputMatch3);
            Assert.True(output.Nodes[0].Output > 0.95);
            output.CalculateOutputs(inputNoMatch1);
            Assert.True(output.Nodes[0].Output < 0.05);
            output.CalculateOutputs(inputNoMatch2);
            Assert.True(output.Nodes[0].Output < 0.05);
            output.CalculateOutputs(inputNoMatch3);
            Assert.True(output.Nodes[0].Output < 0.05);
        }

        [Fact]
        public void TrainFilterToRBGFeatureSortOfWell()
        {
            var r = new Layer1D(4, new Layer[0], ActivationFunctionType.RELU, InitialisationFunctionType.None);
            var g = new Layer1D(4, new Layer[0], ActivationFunctionType.RELU, InitialisationFunctionType.None);
            var b = new Layer1D(4, new Layer[0], ActivationFunctionType.RELU, InitialisationFunctionType.None);
            var filter1 = new Filter1D(new[] { r, g, b }, 2, ActivationFunctionType.RELU, InitialisationFunctionType.GlorotUniform);
            var filter2 = new Filter1D(new[] { r, g, b }, 2, ActivationFunctionType.RELU, InitialisationFunctionType.GlorotUniform);
            var filter3 = new Filter1D(new[] { r, g, b }, 2, ActivationFunctionType.RELU, InitialisationFunctionType.GlorotUniform);
            var output = new Layer(3, new Layer[] { filter1, filter2, filter3 }, ActivationFunctionType.Sigmoid, InitialisationFunctionType.HeEtAl);
            output.AddMomentumRecursively();
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

                output.Backpropagate(inputs, targetOutputs, ErrorFunctionType.CrossEntropy, 0.01, 0.9);
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
        public void TrainNetworkWithPoolingSortOfWellRgb()
        {
            var r = new Layer1D(4, new Layer[0], ActivationFunctionType.RELU, InitialisationFunctionType.None);
            var g = new Layer1D(4, new Layer[0], ActivationFunctionType.RELU, InitialisationFunctionType.None);
            var b = new Layer1D(4, new Layer[0], ActivationFunctionType.RELU, InitialisationFunctionType.None);
            var filters = new[]
            {
                new Filter1D(new[] {r, g, b}, 2, ActivationFunctionType.RELU, InitialisationFunctionType.GlorotUniform),
                new Filter1D(new[] {r, g, b}, 2, ActivationFunctionType.RELU, InitialisationFunctionType.GlorotUniform),
                new Filter1D(new[] {r, g, b}, 2, ActivationFunctionType.RELU, InitialisationFunctionType.GlorotUniform)
            };
            filters.AddPooling(2);
            var output = new Layer(3, filters, ActivationFunctionType.Sigmoid, InitialisationFunctionType.HeEtAl);
            output.AddMomentumRecursively();
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

                output.Backpropagate(inputs, targetOutputs, ErrorFunctionType.CrossEntropy, 0.01, 0.9);
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