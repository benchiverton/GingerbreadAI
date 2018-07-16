using System;
using NeuralNetwork.Data;

namespace NeuralNetwork.Library
{
    using System.Collections.Generic;
    using System.Linq;

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
            foreach (var nodeWeightsKey in node.Weights.Keys.ToList())
            {
                foreach (var nodeKey in node.Weights[nodeWeightsKey].Keys.ToList())
                {
                    node.Weights[nodeWeightsKey][nodeKey] = (double)rand.Next(2000000) / 1000000 - 1;
                }
            }
            var biasWeightKeys = new List<NodeLayer>(node.BiasWeights.Keys.ToList());
            foreach (var biasWeightKey in biasWeightKeys)
            {
                node.BiasWeights[biasWeightKey] = (double)rand.Next(2000000) / 1000000 - 1;
            }
        }

        /// <summary>
        ///     Initialises each Node in Nodes with random weights.
        /// </summary>
        /// <param name="rand"></param>
        /// <param name="nodeGroup"></param>
        public static void Initialise(Random rand, NodeLayer nodeGroup)
        {
            foreach (var node in nodeGroup.Nodes)
            {
                Initialise(rand, node);
            }
            foreach (var nodeGroupPrev in nodeGroup.PreviousGroups)
            {
                if(nodeGroupPrev.PreviousGroups.Length != 0)
                {
                    Initialise(rand, nodeGroupPrev);
                }
            }
        }
    }
}