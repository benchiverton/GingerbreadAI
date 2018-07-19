namespace NeuralNetwork
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using Data;

    public static class LayerInitialiser
    {
        /// <summary>
        ///     Initialises each Node in Nodes with random weights.
        /// </summary>
        /// <param name="rand"></param>
        /// <param name="nodeGroup"></param>
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
        ///     Initialises a Node with random weights.
        /// </summary>
        /// <param name="rand"></param>
        /// <param name="node"></param>
        private static void Initialise(Random rand, Node node)
        {
            if (node == null) return;
            foreach (var prevNode in node.Weights.Keys.ToList())
            {
                node.Weights[prevNode] = (double)rand.Next(2000000) / 1000000 - 1;
            }
            var biasWeightKeys = new List<Layer>(node.BiasWeights.Keys.ToList());
            foreach (var biasWeightKey in biasWeightKeys)
            {
                node.BiasWeights[biasWeightKey] = (double)rand.Next(2000000) / 1000000 - 1;
            }
        }
    }
}