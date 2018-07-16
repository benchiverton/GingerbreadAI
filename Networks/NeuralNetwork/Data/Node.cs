using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Data
{
    /// <summary>
    ///     A class containing the properties and methods that a single node in a network requires.
    /// </summary>
    public class Node
    {
        /// <summary>
        ///     The weights associated with a node. These values correspond to the nodeGroup which
        ///     feed into the nodeGroup containing this node, ie Weights[1][0] => nodeGroupPrev[1].Nodes[0].
        /// </summary>
        public double[][] Weights { get; set; }

        /// <summary>
        ///     The bias that is passed into this node from the previous groupSSS!!
        /// </summary>
        public double[] BiasWeights { get; set; }

        /// <summary>
        ///     Default constructor
        /// </summary>
        public Node()
        {
        }

        /// <summary>
        ///     Constructs a node with the correct amount of weights, given an array of the NodeGroups which feed into this node.
        /// </summary>
        /// <param name="nodeGroupPrev"></param>
        public Node(IReadOnlyList<NodeLayer> nodeGroupPrev)
        {
            Weights = new double[nodeGroupPrev.Count][];
            // each double[] in weights corresponds to the relevant array of nodes in the previous group.
            for (var i = 0; i < nodeGroupPrev.Count; i++)
                Weights[i] = new double[nodeGroupPrev[i].Nodes.Length];
            BiasWeights = new double[nodeGroupPrev.Count];
        }

        public override string ToString()
        {
            var s = new StringBuilder();
            for (var i = 0; i < Weights.Length; i++)
            {
                s.Append($"Node Array {i}:\n");
                for (var j = 0; j < Weights[i].Length; j++)
                    s.Append($"Weight {j}: {Weights[i][j]}\n");
            }
            for (var i = 0; i < BiasWeights.Length; i++)
                s.Append($"Bias {i}: {BiasWeights[i]}\n");
            return s.ToString();
        }
    }
}