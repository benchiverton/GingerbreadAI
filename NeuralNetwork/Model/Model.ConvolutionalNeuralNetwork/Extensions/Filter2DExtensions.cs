using System.Collections.Generic;
using Model.ConvolutionalNeuralNetwork.Models;
using Model.NeuralNetwork.Models;

namespace Model.ConvolutionalNeuralNetwork.Extensions
{
    public static class Filter2DExtensions
    {
        public static void AddPooling(this Filter2D filter, (int height, int width) poolingDimensions)
        {
            var filterWeightMap = new Dictionary<Layer, PooledWeight[,]>();
            foreach (var prevLayer in filter.PreviousLayers)
            {
                // 'catchment' area
                var pooledWeightMap = new PooledWeight[filter.Shape.width + poolingDimensions.width - 1, filter.Shape.height + poolingDimensions.height - 1];
                for (var i = 0; i < poolingDimensions.height; i++) // down
                {
                    for (var j = 0; j < poolingDimensions.width; j++) // across
                    {
                        for (var k = 0; k < filter.Shape.height; k++) // down
                        {
                            for (var l = 0; l < filter.Shape.width; l++) // across
                            {
                                if (pooledWeightMap[j + l, i + k] == null)
                                {
                                    pooledWeightMap[j + l, i + k] = new PooledWeight(poolingDimensions.height * poolingDimensions.width);
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
            for (var i = 0; i < prevLayerDimensions.height - filter.Shape.height - poolingDimensions.height + 2; i += poolingDimensions.height) // down
            {
                for (var j = 0; j < prevLayerDimensions.width - filter.Shape.width - poolingDimensions.width + 2; j += poolingDimensions.width) // across
                {
                    var nodeWeights = new Dictionary<Node, Weight>();
                    for (var k = 0; k < filter.Shape.width + poolingDimensions.height - 1; k++) // down
                    {
                        for (var l = 0; l < filter.Shape.height + poolingDimensions.width - 1; l++) // across
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
