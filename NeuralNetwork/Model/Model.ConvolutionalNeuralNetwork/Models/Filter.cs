using System.Collections.Generic;
using Model.NeuralNetwork.Models;

namespace Model.ConvolutionalNeuralNetwork.Models
{
    public class Filter : Layer
    {
        public Filter(Layer[] previousLayers, int prvLayersHeight, int prvLayersWidth, int filterDimension)
        {
            PreviousLayers = previousLayers;
            
            var nodes = new List<Node>();
            var filterWeightMap = new Dictionary<Layer, Weight[,]>();
            foreach (var prevLayer in previousLayers)
            {
                var filterWeights = new Weight[filterDimension, filterDimension];
                for (var i = 0; i < filterDimension; i++) // across
                {
                    for (var j = 0; j < filterDimension; j++) // down
                    {
                        filterWeights[j, i] = new Weight(0);
                    }
                }
                filterWeightMap.Add(prevLayer, filterWeights);
            }
            for (var i = 0; i < prvLayersHeight - filterDimension + 1; i++)
            {
                for (var j = 0; j < prvLayersWidth - filterDimension + 1; j++)
                {
                    var nodeWeights = new Dictionary<Node, Weight>();
                    for (var k = 0; k < filterDimension; k++) // across
                    {
                        for (var l = 0; l < filterDimension; l++) // down
                        {
                            var nodePosition = j + l + (i + k) * prvLayersWidth;
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
