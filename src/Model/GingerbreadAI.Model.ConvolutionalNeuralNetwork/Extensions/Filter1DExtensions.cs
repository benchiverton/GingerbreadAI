using System.Collections.Generic;
using GingerbreadAI.Model.ConvolutionalNeuralNetwork.Models;
using GingerbreadAI.Model.NeuralNetwork.Models;

namespace GingerbreadAI.Model.ConvolutionalNeuralNetwork.Extensions;

public static class Filter1DExtensions
{
    public static void AddPooling(this Filter1D filter, int poolingDimension)
    {
        var nodes = new List<Node>();

        var dimensions = (filter.PreviousLayers[0] as Layer1D).Size;
        for (var i = 0; i < dimensions - 2; i += poolingDimension)
        {
            var nodesInPool = new List<Node>();
            for (var j = i; j < i + poolingDimension; j++)
            {
                nodesInPool.Add(filter.Nodes[j]);
            }
            nodes.Add(new PooledNode(nodesInPool));
        }

        filter.Nodes = nodes.ToArray();
    }
}
