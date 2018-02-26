using System;
using NeuralNetwork.Data;

namespace NeuralNetwork.Library
{
    public class Initialiser
    {
        /// <summary>
        ///     Initialises this Node with random weights.
        /// </summary>
        /// <param name="rand"></param>
        /// <param name="node"></param>
        public static void Initialise(Random rand, Node node)
        {
            if (node == null) return;
            foreach (var weightArr in node.Weights)
                for (var j = 0; j < weightArr.Length; j++)
                    weightArr[j] = (double) rand.Next(1000000) / 1000000;
            for (var i = 0; i < node.BiasWeights.Length; i++)
                node.BiasWeights[i] = (double) rand.Next(1000000) / 1000000;
        }

        /// <summary>
        ///     Initialises each Node in Nodes with random weights.
        /// </summary>
        /// <param name="rand"></param>
        /// <param name="nodeLayer"></param>
        public static void Initialise(Random rand, NodeLayer nodeLayer)
        {
            foreach (var node in nodeLayer.Nodes)
                Initialise(rand, node);
        }

        /// <summary>
        ///     Initialises each node within the network with random weights.
        /// </summary>
        /// <param name="rand"></param>
        /// <param name="nodeNetwork"></param>
        public static void Initialise(Random rand, NodeNetwork nodeNetwork)
        {
            foreach (var layer in nodeNetwork.Layers)
                Initialise(rand, layer);
        }
    }
}