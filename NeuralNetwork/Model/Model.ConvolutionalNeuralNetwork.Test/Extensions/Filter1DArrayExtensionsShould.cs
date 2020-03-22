using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Model.ConvolutionalNeuralNetwork.Extensions;
using Model.ConvolutionalNeuralNetwork.Models;
using Model.NeuralNetwork.ActivationFunctions;
using Model.NeuralNetwork.Initialisers;
using Model.NeuralNetwork.Models;
using Xunit;

namespace Model.ConvolutionalNeuralNetwork.Test.Extensions
{
    public class Filter1DArrayExtensionsShould
    {
        [Fact]
        public void CorrectlyPoolToSingle()
        {
            // Input:
            // 1  2  3
            //
            // Filter:
            // 1,2 2,3
            //
            // Pooling:
            // 1,2 + 2,3
            var input = new Layer1D(3, new Layer[0], ActivationFunctionType.RELU, InitialisationFunctionType.GlorotUniform);
            var filter = new Filter1D(new[] { input }, 2, ActivationFunctionType.RELU, InitialisationFunctionType.GlorotUniform);
            // Expected _magnitudes: 
            // 0.50: 1,3
            // 1.00: 2
            var expectedWeights = new Dictionary<int, double>
            {
                [0] = 0.5,
                [1] = 1,
                [2] = 0.5,
            };

            filter.AddPooling(2);

            var node = Assert.Single(filter.Nodes);
            Assert.Equal(3, node.Weights.Count);
            for (var i = 0; i < input.Nodes.Length; i++)
            {
                var (_, weight) = node.Weights.First(w => w.Key == input.Nodes[i]);
                // set weight so that we can assert the _magnitude is correctly set
                weight.Value = 1;
                Assert.Equal(expectedWeights[i], weight.Value);
            }
        }
    }
}
