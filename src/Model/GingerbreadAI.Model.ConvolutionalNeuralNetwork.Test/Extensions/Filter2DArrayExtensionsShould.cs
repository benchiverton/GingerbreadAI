using System.Collections.Generic;
using System.Linq;
using GingerbreadAI.Model.ConvolutionalNeuralNetwork.Extensions;
using GingerbreadAI.Model.ConvolutionalNeuralNetwork.Models;
using GingerbreadAI.Model.NeuralNetwork.ActivationFunctions;
using GingerbreadAI.Model.NeuralNetwork.InitialisationFunctions;
using GingerbreadAI.Model.NeuralNetwork.Models;
using Xunit;

namespace GingerbreadAI.Model.ConvolutionalNeuralNetwork.Test.Extensions;

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
        // max(1,2,4,5; 2,3,4,6; 4,5,7,8; 5,6,8,9)
        var input = new Layer2D((3, 3), System.Array.Empty<Layer>(), ActivationFunctionType.RELU, InitialisationFunctionType.GlorotUniform);
        var filter = new Filter2D(new[] { input }, (2, 2), ActivationFunctionType.RELU, InitialisationFunctionType.GlorotUniform);

        filter.AddPooling((2, 2));

        var node = Assert.Single(filter.Nodes);
        var pooledNode = Assert.IsType<PooledNode>(node);
        Assert.Equal(4, pooledNode.UnderlyingNodes.Count);
    }
}
