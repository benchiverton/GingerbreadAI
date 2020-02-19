using System.Collections.Generic;
using Model.ConvolutionalNeuralNetwork.Models;
using Model.NeuralNetwork.Models;

namespace Model.ConvolutionalNeuralNetwork.Extensions
{
    public static class Filter2DExtensions
    {
        public static void AddPooling(this Filter2D filter, int poolingDimension)
        {
            var nodes = new List<Node>();

            var (height, width) = (filter.PreviousLayers[0] as Layer2D).Dimensions;
            for (var i = 0; i < height - filter.Dimension - poolingDimension + 2; i++)
            {
                for (var j = 0; j < width - filter.Dimension - poolingDimension + 2; j++)
                {
                    var nodeWeights = new Dictionary<Node, Weight>();
                    for (var k = 0; k < poolingDimension; k++) // across
                    {
                        for (var l = 0; l < poolingDimension; l++) // down
                        {
                            var nodePosition = j + l + (i + k) * (width - filter.Dimension + 1);
                            var filterNode = filter.Nodes[nodePosition];
                            foreach (var previousNode in filterNode.Weights.Keys)
                            {
                                if (!nodeWeights.ContainsKey(previousNode))
                                {
                                    nodeWeights.Add(previousNode, new Pooled2DWeight(poolingDimension));
                                }
                                else
                                {
                                    (nodeWeights[previousNode] as Pooled2DWeight).IncreaseOccurrences();
                                }
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
