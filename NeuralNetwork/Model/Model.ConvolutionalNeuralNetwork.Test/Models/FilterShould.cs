using Model.ConvolutionalNeuralNetwork.Models;
using Model.NeuralNetwork.Models;
using Xunit;
using Xunit.Abstractions;
using Enumerable = System.Linq.Enumerable;

namespace Model.ConvolutionalNeuralNetwork.Test.Models
{
    public class FilterShould
    {
        private readonly ITestOutputHelper _testOutputHelper;

        public FilterShould(ITestOutputHelper testOutputHelper)
        {
            _testOutputHelper = testOutputHelper;
        }

        [Fact]
        public void ResolveCorrectNodeReferences()
        {
            // inputX:
            // 0  1  2  3  4  5
            // 6  7  8  9  10 11
            // 12 13 14 15 16 17
            // 18 19 20 21 22 23
            // initial filter position:
            // X  X  X  O  O  O
            // X  X  X  O  O  O
            // X  X  X  O  O  O
            // O  O  O  O  O  O
            var input1 = new Layer(24, new Layer[0]);
            var input2 = new Layer(24, new Layer[0]);
            var input3 = new Layer(24, new Layer[0]);

            var filter = new Filter(new[] { input1, input2, input3 }, 4, 6, 3);

            Assert.Equal(8, filter.Nodes.Length);
            for (var i = 0; i < 3; i++)
            {
                for (var j = 0; j < 3; j++)
                {
                    Assert.Contains(input1.Nodes[6 * i + j], Enumerable.ToList(filter.Nodes[0].Weights.Keys));
                    Assert.Contains(input2.Nodes[6 * i + j], Enumerable.ToList(filter.Nodes[0].Weights.Keys));
                    Assert.Contains(input3.Nodes[6 * i + j], Enumerable.ToList(filter.Nodes[0].Weights.Keys));
                }
            }
        }
    }
}