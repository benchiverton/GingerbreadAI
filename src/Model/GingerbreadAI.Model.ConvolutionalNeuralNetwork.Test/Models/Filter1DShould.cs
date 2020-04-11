using GingerbreadAI.Model.ConvolutionalNeuralNetwork.Models;
using GingerbreadAI.Model.NeuralNetwork.ActivationFunctions;
using GingerbreadAI.Model.NeuralNetwork.InitialisationFunctions;
using GingerbreadAI.Model.NeuralNetwork.Models;
using Xunit;
using Xunit.Abstractions;

namespace GingerbreadAI.Model.ConvolutionalNeuralNetwork.Test.Models
{
    public class Filter1DShould
    {
        private readonly ITestOutputHelper _testOutputHelper;

        public Filter1DShould(ITestOutputHelper testOutputHelper)
        {
            _testOutputHelper = testOutputHelper;
        }

        [Fact]
        public void ResolveCorrectNodeReferences()
        {
            // inputX:
            // 0  1  2  3  4  5
            // initial filter position:
            // X  X  X  O  O  O
            var input1 = new Layer1D(6, new Layer[0], ActivationFunctionType.RELU, InitialisationFunctionType.GlorotUniform);
            var input2 = new Layer1D(6, new Layer[0], ActivationFunctionType.RELU, InitialisationFunctionType.GlorotUniform);
            var input3 = new Layer1D(6, new Layer[0], ActivationFunctionType.RELU, InitialisationFunctionType.GlorotUniform);

            var filter = new Filter1D(new[] { input1, input2, input3 }, 3, ActivationFunctionType.RELU, InitialisationFunctionType.GlorotUniform);

            Assert.Equal(4, filter.Nodes.Length);
            for (var i = 0; i < 3; i++)
            {
                Assert.Contains(input1.Nodes[i], filter.Nodes[0].Weights.Keys);
                Assert.Contains(input2.Nodes[i], filter.Nodes[0].Weights.Keys);
                Assert.Contains(input3.Nodes[i], filter.Nodes[0].Weights.Keys);
            }
        }
    }
}
