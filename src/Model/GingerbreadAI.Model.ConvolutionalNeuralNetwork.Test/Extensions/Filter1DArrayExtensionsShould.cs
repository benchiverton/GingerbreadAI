using System;
using GingerbreadAI.Model.ConvolutionalNeuralNetwork.Extensions;
using GingerbreadAI.Model.ConvolutionalNeuralNetwork.Models;
using GingerbreadAI.Model.NeuralNetwork.ActivationFunctions;
using GingerbreadAI.Model.NeuralNetwork.InitialisationFunctions;
using GingerbreadAI.Model.NeuralNetwork.Models;
using Xunit;

namespace GingerbreadAI.Model.ConvolutionalNeuralNetwork.Test.Extensions;

public class Filter1DArrayExtensionsShould
{
    [Fact]
    public void CorrectlyPoolToSingle()
    {
        // Input:
        // 0  1  2
        //
        // Filter:
        // 0,1 1,2
        //
        // Pooling:
        // max(0,1; 1,2)
        var input = new Layer1D(3, Array.Empty<Layer>(), ActivationFunctionType.RELU, InitialisationFunctionType.GlorotUniform);
        var filter = new Filter1D(new[] { input }, 2, ActivationFunctionType.RELU, InitialisationFunctionType.GlorotUniform);

        filter.AddPooling(2);

        var node = Assert.Single(filter.Nodes);
        var pooledNode = Assert.IsType<PooledNode>(node);
        Assert.Equal(2, pooledNode.UnderlyingNodes.Count);
    }
}
