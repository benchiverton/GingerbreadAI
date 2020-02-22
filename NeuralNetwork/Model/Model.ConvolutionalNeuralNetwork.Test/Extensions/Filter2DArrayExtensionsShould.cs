using System.Collections.Generic;
using System.Linq;
using System.Xml;
using Model.ConvolutionalNeuralNetwork.Extensions;
using Model.ConvolutionalNeuralNetwork.Models;
using Model.NeuralNetwork.Models;
using Xunit;

namespace Model.ConvolutionalNeuralNetwork.Test.Extensions
{
    public class Filter2DArrayExtensionsShould
    {
        [Fact]
        public void CorrectlyPoolToSingle()
        {
            // Input:
            // 1  2  3
            // 4  5  6
            // 7  8  9
            //
            // Filter:
            // 1,2,4,5  2,3,4,6
            // 4,5,7,8  5,6,8,9
            //
            // Pooling:
            // 1,2,4,5 + 2,3,4,6 + 4,5,7,8 + 5,6,8,9
            var input = new Layer2D((3, 3), new Layer[0]);
            var filter = new Filter2D(new[] { input }, 2);
            // Expected Weights: 
            // 0.25: 1,3,7,9
            // 0.50: 2,4,6,8
            // 1.00: 5
            var expectedWeights = new Dictionary<int, double>
            {
                [0] = 0.25,
                [1] = 0.5,
                [2] = 0.25,
                [3] = 0.5,
                [4] = 1d,
                [5] = 0.5,
                [6] = 0.25,
                [7] = 0.5,
                [8] = 0.25,
            };

            filter.AddPooling(2);

            var node = Assert.Single(filter.Nodes);
            Assert.Equal(9, node.Weights.Count);
            var enumerator = node.Weights.GetEnumerator();
            var i = 0;
            while (enumerator.MoveNext())
            {
                Assert.Contains(enumerator.Current.Key, input.Nodes);
                // so that we can assert the _magnitude is correctly set
                enumerator.Current.Value.Value = 1;
                Assert.Equal(expectedWeights[i], enumerator.Current.Value.Value);
                i++;
            }
            enumerator.Dispose();
        }
    }
}
