using System.Collections.Generic;
using Model.ConvolutionalNeuralNetwork.Models;
using Model.NeuralNetwork.Models;

namespace Model.ConvolutionalNeuralNetwork.Extensions
{
    public static class Filter2DExtensions
    {
        public static void AddPooling(this Filter2D filter, int poolingDimension)
        {
            var filterWeightMap = new Dictionary<Layer, PooledWeight[,]>();
            foreach (var prevLayer in filter.PreviousLayers)
            {
                // 'catchment' area
                var pooledWeightMap = new PooledWeight[filter.Shape.width + poolingDimension - 1, filter.Shape.height + poolingDimension - 1];
                for (var i = 0; i < poolingDimension; i++) // down
                {
                    for (var j = 0; j < poolingDimension; j++) // across
                    {
                        for (var k = 0; k < filter.Shape.height; k++) // down
                        {
                            for (var l = 0; l < filter.Shape.width; l++) // across
                            {
                                if (pooledWeightMap[j + l, i + k] == null)
                                {
                                    pooledWeightMap[j + l, i + k] = new PooledWeight(2, poolingDimension);
                                }
                                else
                                {
                                    pooledWeightMap[j + l, i + k].IncreaseOccurrences();
                                }
                            }
                        }
                    }
                }
                filterWeightMap.Add(prevLayer, pooledWeightMap);
            }

            var prevLayerDimensions = (filter.PreviousLayers[0] as Layer2D).Shape;
            var nodes = new List<Node>();
            for (var i = 0; i < prevLayerDimensions.height - filter.Shape.height - poolingDimension + 2; i += poolingDimension) // down
            {
                for (var j = 0; j < prevLayerDimensions.width - filter.Shape.width - poolingDimension + 2; j += poolingDimension) // across
                {
                    var nodeWeights = new Dictionary<Node, Weight>();
                    for (var k = 0; k < filter.Shape.width + poolingDimension - 1; k++) // down
                    {
                        for (var l = 0; l < filter.Shape.height + poolingDimension - 1; l++) // across
                        {
                            var nodePosition = j + l + (i + k) * prevLayerDimensions.width;
                            foreach (var previousLayer in filter.PreviousLayers)
                            {
                                nodeWeights.Add(previousLayer.Nodes[nodePosition], filterWeightMap[previousLayer][l, k]);
                            }
                        }
                    }
                    nodes.Add(new Node
                    {
                        Weights = nodeWeights
                    });
                }
            }

            filter.Nodes = nodes.ToArray();
        }
    }
}
