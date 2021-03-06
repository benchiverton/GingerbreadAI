using System.Collections.Generic;
using GingerbreadAI.Model.ConvolutionalNeuralNetwork.Models;
using GingerbreadAI.Model.NeuralNetwork.Models;

namespace GingerbreadAI.Model.ConvolutionalNeuralNetwork.Extensions
{
    public static class Filter2DExtensions
    {
        public static void AddPooling(this Filter2D filter, (int height, int width) poolingDimensions)
        {
            var filterWeightMap = new Dictionary<Layer, WeightWithPooling[,]>();
            foreach (var prevLayer in filter.PreviousLayers)
            {
                // 'catchment' area
                var pooledWeightMap = new WeightWithPooling[filter.Shape.width + poolingDimensions.width - 1, filter.Shape.height + poolingDimensions.height - 1];
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
                                    pooledWeightMap[j + l, i + k] = new WeightWithPooling(poolingDimensions.height * poolingDimensions.width, 0d);
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

            var (height, width) = ((Layer2D) filter.PreviousLayers[0]).Shape;
            var nodes = new List<Node>();
            for (var i = 0; i < height - filter.Shape.height - poolingDimensions.height + 2; i += poolingDimensions.height) // down
            {
                for (var j = 0; j < width - filter.Shape.width - poolingDimensions.width + 2; j += poolingDimensions.width) // across
                {
                    var node = new Node();
                    for (var k = 0; k < (filter.Shape.width + poolingDimensions.height) - 1; k++) // down
                    {
                        for (var l = 0; l < filter.Shape.height + poolingDimensions.width - 1; l++) // across
                        {
                            var nodePosition = j + l + ((i + k) * width);
                            foreach (var previousLayer in filter.PreviousLayers)
                            {
                                node.Weights.Add(previousLayer.Nodes[nodePosition], filterWeightMap[previousLayer][l, k]);
                            }
                        }
                    }
                    nodes.Add(node);
                }
            }

            filter.Nodes = nodes.ToArray();
        }
    }
}
