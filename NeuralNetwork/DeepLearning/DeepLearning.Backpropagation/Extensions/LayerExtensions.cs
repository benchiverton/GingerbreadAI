using System.Collections.Generic;
using Model.NeuralNetwork.Models;

namespace DeepLearning.Backpropagation.Extensions
{
    public static class LayerExtensions
    {
        // builds a copy of the network with different weight references
        public static Layer GenerateMomentum(this Layer layer)
        {
            var momentum = new Layer
            {
                Nodes = new Node[layer.Nodes.Length],
                PreviousLayers = new Layer[layer.PreviousLayers.Length]
            };

            for (var i = 0; i < layer.Nodes.Length; i++)
            {
                var newNode = new Node
                {
                    Weights = new Dictionary<Node, Weight>(),
                    BiasWeights = new Dictionary<Layer, Weight>()
                };

                foreach (var weightKey in layer.Nodes[i].Weights.Keys)
                {
                    newNode.Weights.Add(weightKey, new Weight(0));
                }

                foreach (var biasWeightKey in layer.Nodes[i].BiasWeights.Keys)
                {
                    newNode.BiasWeights.Add(biasWeightKey, new Weight(0));
                }

                momentum.Nodes[i] = newNode;
            }

            for (var i = 0; i < layer.PreviousLayers.Length; i++)
            {
                momentum.PreviousLayers[i] = layer.PreviousLayers[i].GenerateMomentum();
            }

            return momentum;
        }
    }
}
