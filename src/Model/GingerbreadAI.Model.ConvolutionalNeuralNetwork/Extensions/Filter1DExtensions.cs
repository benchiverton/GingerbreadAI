using System.Collections.Generic;
using GingerbreadAI.Model.ConvolutionalNeuralNetwork.Models;
using GingerbreadAI.Model.NeuralNetwork.Models;

namespace GingerbreadAI.Model.ConvolutionalNeuralNetwork.Extensions;

public static class Filter1DExtensions
{
    public static void AddPooling(this Filter1D filter, int poolingDimension)
    {
        var filterWeightMap = new Dictionary<Layer, WeightWithPooling[]>();
        foreach (var prevLayer in filter.PreviousLayers)
        {
            // 'catchment' area
            var pooledWeightMap = new WeightWithPooling[filter.Size + poolingDimension - 1];
            for (var i = 0; i < poolingDimension; i++)
            {
                for (var j = 0; j < filter.Size; j++)
                {
                    if (pooledWeightMap[i + j] == null)
                    {
                        pooledWeightMap[i + j] = new WeightWithPooling(poolingDimension, 0d);
                    }
                    else
                    {
                        pooledWeightMap[i + j].IncreaseOccurrences();
                    }
                }
            }
            filterWeightMap.Add(prevLayer, pooledWeightMap);
        }

        var prevLayerSize = ((Layer1D)filter.PreviousLayers[0]).Size;
        var nodes = new List<Node>();
        for (var i = 0; i < prevLayerSize - filter.Size - poolingDimension + 2; i += poolingDimension)
        {
            var node = new Node();
            for (var j = 0; j < filter.Size + poolingDimension - 1; j++)
            {
                var nodePosition = i + j;
                foreach (var previousLayer in filter.PreviousLayers)
                {
                    node.Weights.Add(previousLayer.Nodes[nodePosition], filterWeightMap[previousLayer][j]);
                }
            }
            nodes.Add(node);
        }

        filter.Nodes = nodes.ToArray();
    }
}
