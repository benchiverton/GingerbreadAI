using System.Collections.Generic;
using Model.NeuralNetwork.Models;

namespace Model.ConvolutionalNeuralNetwork.Models
{
    public class Filter2D : Layer
    {
        public int Dimension { get; }

        public Filter2D(Layer2D[] previousLayers, int dimension)
        {
            PreviousLayers = previousLayers;
            Dimension = dimension;

            var (height, width) = previousLayers[0].Dimensions;
            var nodes = new List<Node>();
            var filterWeightMap = new Dictionary<Layer, Weight[,]>();
            foreach (var prevLayer in previousLayers)
            {
                var filterWeights = new Weight[Dimension, Dimension];
                for (var i = 0; i < Dimension; i++) // across
                {
                    for (var j = 0; j < Dimension; j++) // down
                    {
                        filterWeights[j, i] = new Weight(0);
                    }
                }
                filterWeightMap.Add(prevLayer, filterWeights);
            }
            for (var i = 0; i < height - Dimension + 1; i++)
            {
                for (var j = 0; j < width - Dimension + 1; j++)
                {
                    var nodeWeights = new Dictionary<Node, Weight>();
                    for (var k = 0; k < Dimension; k++) // across
                    {
                        for (var l = 0; l < Dimension; l++) // down
                        {
                            var nodePosition = j + l + (i + k) * width;
                            foreach (var prevLayer in previousLayers)
                            {
                                nodeWeights.Add(prevLayer.Nodes[nodePosition], filterWeightMap[prevLayer][l, k]);
                            }
                        }
                    }
                    nodes.Add(new Node
                    {
                        Weights = nodeWeights
                    });
                }
            }

            Nodes = nodes.ToArray();
        }
    }
}
