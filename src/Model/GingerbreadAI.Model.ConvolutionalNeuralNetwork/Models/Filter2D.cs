using System.Collections.Generic;
using GingerbreadAI.Model.NeuralNetwork.ActivationFunctions;
using GingerbreadAI.Model.NeuralNetwork.InitialisationFunctions;
using GingerbreadAI.Model.NeuralNetwork.Models;

namespace GingerbreadAI.Model.ConvolutionalNeuralNetwork.Models
{
    public class Filter2D : Layer2D
    {
        public Filter2D(Layer2D[] previousLayers, (int height, int width) filterShape, ActivationFunctionType activationFunctionType, InitialisationFunctionType initialisationFunctionTyp) 
            : base(filterShape, previousLayers, activationFunctionType, initialisationFunctionTyp)
        {
            var filterWeightMap = new Dictionary<Layer, Weight[,]>();
            foreach (var prevLayer in previousLayers)
            {
                var filterWeights = new Weight[Shape.width, Shape.height];
                for (var i = 0; i < Shape.height; i++) // down
                {
                    for (var j = 0; j < Shape.height; j++) // across
                    {
                        filterWeights[j, i] = new Weight(0);
                    }
                }
                filterWeightMap.Add(prevLayer, filterWeights);
            }

            var (height, width) = previousLayers[0].Shape;
            var nodes = new List<Node>();
            for (var i = 0; i < height - Shape.height + 1; i++) // down
            {
                for (var j = 0; j < width - Shape.width + 1; j++) // across
                {
                    var node = new Node();
                    for (var k = 0; k < Shape.height; k++) // down
                    {
                        for (var l = 0; l < Shape.width; l++) // across
                        {
                            var nodePosition = j + l + ((i + k) * width);
                            foreach (var prevLayer in previousLayers)
                            {
                                node.Weights.Add(prevLayer.Nodes[nodePosition], filterWeightMap[prevLayer][l, k]);
                            }
                        }
                    }
                    nodes.Add(node);
                }
            }

            Nodes = nodes.ToArray();
        }
    }
}
