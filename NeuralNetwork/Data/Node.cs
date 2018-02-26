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
        ///     The weights associated with a node. These values correspond to the nodeLayer which
        ///     feed into the nodeLayer containing this node, ie Weights[1][0] => nodeLayerPrev[1].Nodes[0].
        /// </summary>
        public double[][] Weights { get; set; }

        /// <summary>
        ///     The bias that is passed into this node from the previous layerSSS!!
        /// </summary>
        public double[] BiasWeights { get; set; }

        public Node()
        {            
        }

        /// <summary>
        ///     Constructs a node with the correct amount of weights, given an array of the NodeLayers which feed into this node.
        /// </summary>
        /// <param name="nodeLayerPrev"></param>
        public Node(IReadOnlyList<NodeLayer> nodeLayerPrev)
        {
            Weights = new double[nodeLayerPrev.Count][];
            // each double[] in weights corresponds to the relevant array of nodes in the previous layer.
            for (var i = 0; i < nodeLayerPrev.Count; i++)
                Weights[i] = new double[nodeLayerPrev[i].Nodes.Length];
            BiasWeights = new double[nodeLayerPrev.Count];
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