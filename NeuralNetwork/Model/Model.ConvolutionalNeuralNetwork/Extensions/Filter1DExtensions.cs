using System;
using System.Collections.Generic;
using System.Text;
using Model.ConvolutionalNeuralNetwork.Models;
using Model.NeuralNetwork.Models;

namespace Model.ConvolutionalNeuralNetwork.Extensions
{
    public static class Filter1DExtensions
    {
        public static void AddPooling(this Filter1D filter, int poolingDimension)
        {
            var filterWeightMap = new Dictionary<Layer, PooledWeight[]>();
            foreach (var prevLayer in filter.PreviousLayers)
            {
                // 'catchment' area
                var pooledWeightMap = new PooledWeight[filter.Size + poolingDimension - 1];
                for (var i = 0; i < poolingDimension; i++)
                {
                    for (var j = 0; j < filter.Size; j++)
                    {
                        if (pooledWeightMap[i + j] == null)
                        {
                            pooledWeightMap[i + j] = new PooledWeight(1, poolingDimension);
                        }
                        else
                        {
                            pooledWeightMap[i + j].IncreaseOccurrences();
                        }
                    }
                }
                filterWeightMap.Add(prevLayer, pooledWeightMap);
            }

            var prevLayerSize = (filter.PreviousLayers[0] as Layer1D).Size;
            var nodes = new List<Node>();
            for (var i = 0; i < prevLayerSize - filter.Size - poolingDimension + 2; i += poolingDimension)
            {
                var nodeWeights = new Dictionary<Node, Weight>();
                for (var j = 0; j < filter.Size + poolingDimension - 1; j++)
                {
                    var nodePosition = i + j;
                    foreach (var previousLayer in filter.PreviousLayers)
                    {
                        nodeWeights.Add(previousLayer.Nodes[nodePosition], filterWeightMap[previousLayer][j]);
                    }
                }
                nodes.Add(new Node
                {
                    Weights = nodeWeights
                });
            }

            filter.Nodes = nodes.ToArray();
        }
    }
}
