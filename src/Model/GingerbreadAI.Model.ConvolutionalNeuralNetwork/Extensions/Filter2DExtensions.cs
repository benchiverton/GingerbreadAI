using System.Collections.Generic;
using GingerbreadAI.Model.ConvolutionalNeuralNetwork.Models;
using GingerbreadAI.Model.NeuralNetwork.Models;

namespace GingerbreadAI.Model.ConvolutionalNeuralNetwork.Extensions;

public static class Filter2DExtensions
{
    public static void AddPooling(this Filter2D filter, (int height, int width) poolingDimensions)
    {
        var nodes = new List<Node>();

        var dimensions = (filter.PreviousLayers[0] as Layer2D).Shape;
        for (var i = 0; i < dimensions.height - 2; i += poolingDimensions.height) // down
        {
            for (var j = 0; j < dimensions.width - 2; j += poolingDimensions.width) // across
            {
                var underlyingNodes = new List<Node>();
                for (var k = 0; k < poolingDimensions.height; k++) // down
                {
                    for (var l = 0; l < poolingDimensions.width; l++) // across
                    {
                        underlyingNodes.Add(filter.Nodes[j + l + ((i + k) * poolingDimensions.width)]);
                    }
                }
                nodes.Add(new PooledNode(underlyingNodes));
            }
        }

        filter.Nodes = nodes.ToArray();
    }
}
