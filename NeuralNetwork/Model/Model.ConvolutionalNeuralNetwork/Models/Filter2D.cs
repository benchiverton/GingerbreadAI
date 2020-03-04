using System.Collections.Generic;
using Model.NeuralNetwork.ActivationFunctions;
using Model.NeuralNetwork.Initialisers;
using Model.NeuralNetwork.Models;

namespace Model.ConvolutionalNeuralNetwork.Models
{
    public class Filter2D : Layer2D
    {
        public Filter2D(Layer2D[] previousLayers, int filterDimension, ActivationFunctionType activationFunctionType, InitialisationFunctionType initialisationFunctionTyp) 
            : base((filterDimension, filterDimension), previousLayers, activationFunctionType, initialisationFunctionTyp)
        {
            ActivationFunctionType = ActivationFunctionType.RELU;
            InitialisationFunctionType = InitialisationFunctionType.Uniform;

            var filterWeightMap = new Dictionary<Layer, Weight[,]>();
            foreach (var prevLayer in previousLayers)
            {
                var filterWeights = new Weight[Dimensions.width, Dimensions.height];
                for (var i = 0; i < Dimensions.height; i++) // down
                {
                    for (var j = 0; j < Dimensions.height; j++) // across
                    {
                        filterWeights[j, i] = new Weight(0);
                    }
                }
                filterWeightMap.Add(prevLayer, filterWeights);
            }

            var prevLayerDimensions = previousLayers[0].Dimensions;
            var nodes = new List<Node>();
            for (var i = 0; i < prevLayerDimensions.height - Dimensions.height + 1; i++) // down
            {
                for (var j = 0; j < prevLayerDimensions.width - Dimensions.width + 1; j++) // across
                {
                    var nodeWeights = new Dictionary<Node, Weight>();
                    for (var k = 0; k < Dimensions.height; k++) // down
                    {
                        for (var l = 0; l < Dimensions.width; l++) // across
                        {
                            var nodePosition = j + l + (i + k) * prevLayerDimensions.width;
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
