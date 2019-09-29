namespace NeuralNetwork
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using AI.Calculations;
    using NeuralNetwork.Models;

    public static class LayerInitialiser
    {
        /// <summary>
        ///     Initialises each Node in Nodes with random weights.
        /// </summary>
        public static void Initialise(Random rand, Layer nodeGroup)
        {
            foreach (var node in nodeGroup.Nodes)
            {
                Initialise(rand, node);
            }
            foreach (var nodeGroupPrev in nodeGroup.PreviousLayers)
            {
                Initialise(rand, nodeGroupPrev);
            }
        }

        /// <summary>
        ///     Initialises a Node with random weights (using He-et-al Initialization).
        /// </summary>
        private static void Initialise(Random rand, Node node)
        {
            if (node == null) return;
            var feedingNodes = node.Weights.Count;
            foreach (var prevNode in node.Weights.Keys.ToList())
            {
                node.Weights[prevNode].Value = NetworkCalculations.GetWeightedInitialisation(rand, feedingNodes);
            }
            var biasWeightKeys = new List<Layer>(node.BiasWeights.Keys.ToList());
            foreach (var biasWeightKey in biasWeightKeys)
            {
                node.BiasWeights[biasWeightKey].Value = NetworkCalculations.GetWeightedInitialisation(rand, feedingNodes);
            }
        }
    }
}