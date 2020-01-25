using System;
using System.Collections.Generic;
using Model.ConvolutionalNeuralNetwork.Models;
using Model.NeuralNetwork;
using Model.NeuralNetwork.Models;
using Xunit;

namespace DeepLearning.Backpropagation.CNN.Test
{
    public class BackpropagationShould
    {
        [Fact]
        public void TrainConvolutionalNetworksSortofWellRgb()
        {
            var r = new Layer(16, new Layer[0]);
            var g = new Layer(16, new Layer[0]);
            var b = new Layer(16, new Layer[0]);
            var filter = new Filter(new[] { r, g, b }, 4, 4, 3);
            var output = new Layer(3, new Layer[] { filter });
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