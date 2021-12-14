using GingerbreadAI.Model.ConvolutionalNeuralNetwork.Models;
using GingerbreadAI.Model.NeuralNetwork.ActivationFunctions;
using GingerbreadAI.Model.NeuralNetwork.InitialisationFunctions;
using GingerbreadAI.Model.NeuralNetwork.Models;
using Xunit;
using Xunit.Abstractions;

namespace GingerbreadAI.Model.ConvolutionalNeuralNetwork.Test.Models;

public class Filter2DShould
{
    private readonly ITestOutputHelper _testOutputHelper;

    public Filter2DShould(ITestOutputHelper testOutputHelper) => _testOutputHelper = testOutputHelper;

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
        var input1 = new Layer2D((4, 6), System.Array.Empty<Layer>(), ActivationFunctionType.RELU, InitialisationFunctionType.GlorotUniform);
        var input2 = new Layer2D((4, 6), System.Array.Empty<Layer>(), ActivationFunctionType.RELU, InitialisationFunctionType.GlorotUniform);
        var input3 = new Layer2D((4, 6), System.Array.Empty<Layer>(), ActivationFunctionType.RELU, InitialisationFunctionType.GlorotUniform);

        var filter = new Filter2D(new[] { input1, input2, input3 }, (3, 3), ActivationFunctionType.RELU, InitialisationFunctionType.GlorotUniform);

        Assert.Equal(8, filter.Nodes.Count);
        for (var i = 0; i < 3; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                Assert.Contains(input1.Nodes[(6 * i) + j], filter.Nodes[0].Weights.Keys);
                Assert.Contains(input2.Nodes[(6 * i) + j], filter.Nodes[0].Weights.Keys);
                Assert.Contains(input3.Nodes[(6 * i) + j], filter.Nodes[0].Weights.Keys);
            }
        }
    }
}
