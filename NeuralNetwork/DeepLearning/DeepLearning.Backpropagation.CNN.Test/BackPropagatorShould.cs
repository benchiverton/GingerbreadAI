using System;
using System.Collections.Generic;
using System.Linq;
using Model.ConvolutionalNeuralNetwork.Extensions;
using Model.ConvolutionalNeuralNetwork.Models;
using Model.NeuralNetwork;
using Model.NeuralNetwork.Models;
using Xunit;
using Xunit.Abstractions;

namespace DeepLearning.Backpropagation.CNN.Test
{
    public class BackpropagationShould
    {
        private readonly ITestOutputHelper _testOutputHelper;

        public BackpropagationShould(ITestOutputHelper testOutputHelper)
        {
            _testOutputHelper = testOutputHelper;
        }

        [Fact]
        public void TrainFilterToFeatureSortOfWell()
        {
            var inputLayer = new Layer2D((3, 3), new Layer[0]);
            var filter = new Filter2D(new[] { inputLayer }, 3);
            var output = new Layer(1, new Layer[] { filter });
            output.Initialise(new Random());
            var inputMatch = new double[] { 1, 0, 1, 0, 1, 0, 1, 0, 1 };
            var inputNoMatch1 = new double[] { 0, 1, 0, 1, 0, 1, 0, 1, 0 };
            var inputNoMatch2 = new double[] { 1, 1, 1, 1, 1, 1, 1, 1, 1 };
            var inputNoMatch3 = new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0 };

            for (var i = 0; i < 10000; i++)
            {
                output.Backpropagate(inputMatch, new double[] { 1 }, 0.5);
                output.Backpropagate(inputNoMatch1, new double[] { 0 }, 0.5);
                output.Backpropagate(inputNoMatch2, new double[] { 0 }, 0.5);
                output.Backpropagate(inputNoMatch3, new double[] { 0 }, 0.5);
            }

            // filter weights should look as follows:
            // +  -  +      -  +  -
            // -  +  -  OR  +  -  +  (Depending on output weight being +ve/-ve)
            // +  -  +      -  +  -
            _testOutputHelper.WriteLine($"filter: {string.Join(",", filter.Nodes[0].Weights.Values.Select(v => v.Value.ToString("0.00")))}");

            var outputMultiplier = output.Nodes[0].Weights[filter.Nodes[0]].Value > 0 ? 1 : -1;
            Assert.True(outputMultiplier * filter.Nodes[0].Weights[inputLayer.Nodes[0]].Value > 0);
            Assert.True(outputMultiplier * filter.Nodes[0].Weights[inputLayer.Nodes[1]].Value < 0);
            Assert.True(outputMultiplier * filter.Nodes[0].Weights[inputLayer.Nodes[2]].Value > 0);
            Assert.True(outputMultiplier * filter.Nodes[0].Weights[inputLayer.Nodes[3]].Value < 0);
            Assert.True(outputMultiplier * filter.Nodes[0].Weights[inputLayer.Nodes[4]].Value > 0);
            Assert.True(outputMultiplier * filter.Nodes[0].Weights[inputLayer.Nodes[5]].Value < 0);
            Assert.True(outputMultiplier * filter.Nodes[0].Weights[inputLayer.Nodes[6]].Value > 0);
            Assert.True(outputMultiplier * filter.Nodes[0].Weights[inputLayer.Nodes[7]].Value < 0);
            Assert.True(outputMultiplier * filter.Nodes[0].Weights[inputLayer.Nodes[8]].Value > 0);
        }

        [Fact]
        public void TrainFilterToMultipleFeaturesSortOfWell()
        {
            var inputLayer = new Layer2D((3, 3), new Layer[0]);
            var filter1 = new Filter2D(new[] { inputLayer }, 3);
            var filter2 = new Filter2D(new[] { inputLayer }, 3);
            var output = new Layer(2, new Layer[] { filter1, filter2 });
            output.Initialise(new Random());
            var fullMatch = new double[] { 1, 1, 0, 1, 0, 1, 0, 1, 1 };
            var inputMatch1 = new double[] { 1, 1, 0, 1, 0, 0, 0, 0, 0 };
            var inputMatch2 = new double[] { 0, 0, 0, 0, 0, 1, 0, 1, 1 };
            var inputNoMatch1 = new double[] { 1, 0, 1, 0, 1, 0, 1, 0, 1 };
            var inputNoMatch2 = new double[] { 0, 0, 1, 0, 1, 0, 1, 0, 0 };

            for (var i = 0; i < 10000; i++)
            {
                output.Backpropagate(fullMatch, new double[] { 1, 1 }, 0.5);
                output.Backpropagate(inputMatch1, new double[] { 1, 0 }, 0.5);
                output.Backpropagate(inputMatch2, new double[] { 0, 1 }, 0.5);
                output.Backpropagate(inputNoMatch1, new double[] { 0, 0 }, 0.5);
                output.Backpropagate(inputNoMatch2, new double[] { 0, 0 }, 0.5);
            }

            // filter weights should look as follows:
            // +  +  -      -  -  +
            // +  -  -  OR  -  +  +  (Depending on output weight being +ve/-ve)
            // -  -  -      +  +  +
            // and
            // -  -  -      +  +  +
            // -  -  +  OR  +  +  -  (Depending on output weight being +ve/-ve)
            // -  +  +      +  -  -
            var filterTopLeft = filter1.Nodes[0].Weights[inputLayer.Nodes[1]].Value * filter1.Nodes[0].Weights[inputLayer.Nodes[2]].Value < 0
                ? filter1
                : filter2;
            var filterBottomRight = filterTopLeft != filter1
                ? filter1
                : filter2;

            _testOutputHelper.WriteLine($"filter top left: {string.Join(",", filterTopLeft.Nodes[0].Weights.Values.Select(v => v.Value.ToString("0.00")))}");
            _testOutputHelper.WriteLine($"filter bottom right: {string.Join(",", filterBottomRight.Nodes[0].Weights.Values.Select(v => v.Value.ToString("0.00")))}");

            // top left
            var outputMultiplierTopLeft = output.Nodes[0].Weights[filterTopLeft.Nodes[0]].Value > 0 ? 1 : -1;
            Assert.True(outputMultiplierTopLeft * filterTopLeft.Nodes[0].Weights[inputLayer.Nodes[0]].Value > 0);
            Assert.True(outputMultiplierTopLeft * filterTopLeft.Nodes[0].Weights[inputLayer.Nodes[1]].Value > 0);
            Assert.True(outputMultiplierTopLeft * filterTopLeft.Nodes[0].Weights[inputLayer.Nodes[2]].Value < 0);
            Assert.True(outputMultiplierTopLeft * filterTopLeft.Nodes[0].Weights[inputLayer.Nodes[3]].Value > 0);
            Assert.True(outputMultiplierTopLeft * filterTopLeft.Nodes[0].Weights[inputLayer.Nodes[4]].Value < 0);
            Assert.True(outputMultiplierTopLeft * filterTopLeft.Nodes[0].Weights[inputLayer.Nodes[5]].Value < 0);
            Assert.True(outputMultiplierTopLeft * filterTopLeft.Nodes[0].Weights[inputLayer.Nodes[6]].Value < 0);
            Assert.True(outputMultiplierTopLeft * filterTopLeft.Nodes[0].Weights[inputLayer.Nodes[7]].Value < 0);
            Assert.True(outputMultiplierTopLeft * filterTopLeft.Nodes[0].Weights[inputLayer.Nodes[8]].Value < 0);
            // bottom right
            var outputMultiplierBottomRight = output.Nodes[1].Weights[filterBottomRight.Nodes[0]].Value > 0 ? 1 : -1;
            Assert.True(outputMultiplierBottomRight * filterBottomRight.Nodes[0].Weights[inputLayer.Nodes[0]].Value < 0);
            Assert.True(outputMultiplierBottomRight * filterBottomRight.Nodes[0].Weights[inputLayer.Nodes[1]].Value < 0);
            Assert.True(outputMultiplierBottomRight * filterBottomRight.Nodes[0].Weights[inputLayer.Nodes[2]].Value < 0);
            Assert.True(outputMultiplierBottomRight * filterBottomRight.Nodes[0].Weights[inputLayer.Nodes[3]].Value < 0);
            Assert.True(outputMultiplierBottomRight * filterBottomRight.Nodes[0].Weights[inputLayer.Nodes[4]].Value < 0);
            Assert.True(outputMultiplierBottomRight * filterBottomRight.Nodes[0].Weights[inputLayer.Nodes[5]].Value > 0);
            Assert.True(outputMultiplierBottomRight * filterBottomRight.Nodes[0].Weights[inputLayer.Nodes[6]].Value < 0);
            Assert.True(outputMultiplierBottomRight * filterBottomRight.Nodes[0].Weights[inputLayer.Nodes[7]].Value > 0);
            Assert.True(outputMultiplierBottomRight * filterBottomRight.Nodes[0].Weights[inputLayer.Nodes[8]].Value > 0);
        }

        [Fact]
        public void TrainConvolutionalNetworksSortofWellWithProxyRgb()
        {
            var r = new Layer2D((4, 4), new Layer[0]);
            var g = new Layer2D((4, 4), new Layer[0]);
            var b = new Layer2D((4, 4), new Layer[0]);
            var proxy1 = new Layer2D((4, 4), new[] { r });
            var proxy2 = new Layer2D((4, 4), new[] { g });
            var proxy3 = new Layer2D((4, 4), new[] { b });
            var filter1 = new Filter2D(new[] { proxy1, proxy2, proxy3 }, 2);
            var filter2 = new Filter2D(new[] { proxy1, proxy2, proxy3 }, 2);
            var filter3 = new Filter2D(new[] { proxy1, proxy2, proxy3 }, 2);
            var output = new Layer(3, new Layer[] { filter1, filter2, filter3 });
            output.Initialise(new Random());
            Dictionary<Layer, double[]> ResolveInputs(bool isRed, bool isGreen, bool isBlue)
            {
                var rInput = isRed
                    ? new double[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }
                    : new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
                var gInput = isGreen
                    ? new double[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }
                    : new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
                var bInput = isBlue
                    ? new double[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }
                    : new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
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
            output.PopulateAllOutputs(redInput);
            Assert.True(output.Nodes[0].Output > 0.95);
            Assert.True(output.Nodes[1].Output < 0.05);
            Assert.True(output.Nodes[2].Output < 0.05);
            var greenInput = ResolveInputs(false, true, false);
            output.PopulateAllOutputs(greenInput);
            Assert.True(output.Nodes[0].Output < 0.05);
            Assert.True(output.Nodes[1].Output > 0.95);
            Assert.True(output.Nodes[2].Output < 0.05);
            var blueInput = ResolveInputs(false, false, true);
            output.PopulateAllOutputs(blueInput);
            Assert.True(output.Nodes[0].Output < 0.05);
            Assert.True(output.Nodes[1].Output < 0.05);
            Assert.True(output.Nodes[2].Output > 0.95);
        }


        [Fact]
        public void TrainConvolutionalNetworksSortofWellRgb()
        {
            var r = new Layer2D((4, 4), new Layer[0]);
            var g = new Layer2D((4, 4), new Layer[0]);
            var b = new Layer2D((4, 4), new Layer[0]);
            var filter1 = new Filter2D(new[] { r, g, b }, 2);
            var filter2 = new Filter2D(new[] { r, g, b }, 2);
            var filter3 = new Filter2D(new[] { r, g, b }, 2);
            var output = new Layer(3, new Layer[] { filter1, filter2, filter3 });
            output.Initialise(new Random());
            Dictionary<Layer, double[]> ResolveInputs(bool isRed, bool isGreen, bool isBlue)
            {
                var rInput = isRed
                    ? new double[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }
                    : new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
                var gInput = isGreen
                    ? new double[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }
                    : new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
                var bInput = isBlue
                    ? new double[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }
                    : new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
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
            output.PopulateAllOutputs(redInput);
            Assert.True(output.Nodes[0].Output > 0.95);
            Assert.True(output.Nodes[1].Output < 0.05);
            Assert.True(output.Nodes[2].Output < 0.05);
            var greenInput = ResolveInputs(false, true, false);
            output.PopulateAllOutputs(greenInput);
            Assert.True(output.Nodes[0].Output < 0.05);
            Assert.True(output.Nodes[1].Output > 0.95);
            Assert.True(output.Nodes[2].Output < 0.05);
            var blueInput = ResolveInputs(false, false, true);
            output.PopulateAllOutputs(blueInput);
            Assert.True(output.Nodes[0].Output < 0.05);
            Assert.True(output.Nodes[1].Output < 0.05);
            Assert.True(output.Nodes[2].Output > 0.95);
        }


        [Fact]
        public void TrainConvolutionalNetworksWithFilterSortofWellRgb()
        {
            var r = new Layer2D((4, 4), new Layer[0]);
            var g = new Layer2D((4, 4), new Layer[0]);
            var b = new Layer2D((4, 4), new Layer[0]);
            var filter1 = new Filter2D(new[] { r, g, b }, 2);
            var filter2 = new Filter2D(new[] { r, g, b }, 2);
            var filter3 = new Filter2D(new[] { r, g, b }, 2);
            var pooling = (new[] { filter1, filter2, filter3 }).AddPooling(2);
            var output = new Layer(3, pooling.ToArray());
            output.Initialise(new Random());
            Dictionary<Layer, double[]> ResolveInputs(bool isRed, bool isGreen, bool isBlue)
            {
                var rInput = isRed
                    ? new double[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }
                    : new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
                var gInput = isGreen
                    ? new double[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }
                    : new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
                var bInput = isBlue
                    ? new double[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }
                    : new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
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
            output.PopulateAllOutputs(redInput);
            Assert.True(output.Nodes[0].Output > 0.95);
            Assert.True(output.Nodes[1].Output < 0.05);
            Assert.True(output.Nodes[2].Output < 0.05);
            var greenInput = ResolveInputs(false, true, false);
            output.PopulateAllOutputs(greenInput);
            Assert.True(output.Nodes[0].Output < 0.05);
            Assert.True(output.Nodes[1].Output > 0.95);
            Assert.True(output.Nodes[2].Output < 0.05);
            var blueInput = ResolveInputs(false, false, true);
            output.PopulateAllOutputs(blueInput);
            Assert.True(output.Nodes[0].Output < 0.05);
            Assert.True(output.Nodes[1].Output < 0.05);
            Assert.True(output.Nodes[2].Output > 0.95);
        }
    }
}