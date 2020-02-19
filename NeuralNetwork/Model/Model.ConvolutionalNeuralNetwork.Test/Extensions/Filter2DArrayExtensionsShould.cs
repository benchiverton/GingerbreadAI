using System.Linq;
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
            var input = new Layer2D((3,3), new Layer[0]);
            var filter = new Filter2D(new [] { input }, 2);

            var pooling = new Pool2D(filter, 2);

            var node = Assert.Single(pooling.Nodes);
            Assert.Equal(4, node.Weights.Count);
            foreach (var nodeKey in node.Weights.Keys)
            {
                Assert.Contains(nodeKey, filter.Nodes);
            }
        }
    }
}
