using System.Collections.Generic;
using GingerbreadAI.Model.NeuralNetwork.ActivationFunctions;
using GingerbreadAI.Model.NeuralNetwork.InitialisationFunctions;
using GingerbreadAI.Model.NeuralNetwork.Models;

namespace GingerbreadAI.Model.ConvolutionalNeuralNetwork.Models
{
    public class Filter1D : Layer1D
    {
        public Filter1D(Layer1D[] previousLayers, int filterSize, ActivationFunctionType activationFunctionType, InitialisationFunctionType initialisationFunctionTyp)
            : base(filterSize, previousLayers, activationFunctionType, initialisationFunctionTyp)
        {
            var filterWeightMap = new Dictionary<Layer, Weight[]>();
            foreach (var prevLayer in previousLayers)
            {
                var filterWeights = new Weight[filterSize];
                for (var i = 0; i < filterSize; i++)
                {
                    filterWeights[i] = new Weight(0);
                }
                filterWeightMap.Add(prevLayer, filterWeights);
            }

            var prevLayerNodesLength = previousLayers[0].Nodes.Count;
            var nodes = new List<Node>();
            for (var i = 0; i < prevLayerNodesLength - filterSize + 1; i++)
            {
                var node = new Node();
                for (var j = 0; j < filterSize; j++)
                {
                    var nodePosition = i + j;
                    foreach (var prevLayer in previousLayers)
                    {
                        node.Weights.Add(prevLayer.Nodes[nodePosition], filterWeightMap[prevLayer][j]);
                    }
                }
                nodes.Add(node);
            }

            Nodes = nodes.ToArray();
        }
    }
}
