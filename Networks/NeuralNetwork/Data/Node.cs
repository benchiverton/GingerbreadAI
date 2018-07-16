using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Data
{
    using System.Linq;

    /// <summary>
    ///     A class containing the properties and methods that a single node in a network requires.
    /// </summary>
    public class Node
    {
        /// <summary>
        ///     The weights associated with a node. These values correspond to the nodeGroup which
        ///     feed into the nodeGroup containing this node, ie Weights[1][0] => nodeGroupPrev[1].Nodes[0].
        /// </summary>
        public Dictionary<NodeLayer, Dictionary<Node, double>> Weights { get; set; }

        /// <summary>
        ///     The bias that is passed into this node from the previous groupSSS!!
        /// </summary>
        public Dictionary<NodeLayer, double> BiasWeights;

        public Node()
        {
        }

        /// <summary>
        ///     Constructs a node with the correct amount of weights, given an array of the NodeGroups which feed into this node.
        /// </summary>
        /// <param name="nodeGroupPrev"></param>
        public Node(IReadOnlyList<NodeLayer> nodeGroupPrev)
        {
            Weights = new Dictionary<NodeLayer, Dictionary<Node, double>>();
            foreach (var prevNodeLayer in nodeGroupPrev)
            {
                var correspondingNodeWeights = new Dictionary<Node, double>();
                foreach (var node in prevNodeLayer.Nodes)
                {
                    correspondingNodeWeights.Add(node, 0);
                }
                Weights.Add(prevNodeLayer, correspondingNodeWeights);
            }

            BiasWeights = new Dictionary<NodeLayer, double>();
            foreach (var prevNodeLayer in nodeGroupPrev)
            {
                BiasWeights.Add(prevNodeLayer, 0);
            }
        }

        public override string ToString()
        {
            var s = new StringBuilder();
            foreach(var layerWeightKey in Weights.Keys.ToList())
            {
                s.Append($"Node Layer {layerWeightKey.Name}:\n");
                foreach (var nodeWeightKey in Weights[layerWeightKey].Keys.ToList())
                {
                    s.Append($"Weight: {Weights[layerWeightKey][nodeWeightKey]}");
                }
            }

            var biasWeights = BiasWeights.Values.ToArray();
            for (var i = 0; i < BiasWeights.Count; i++)
            {
                s.Append($"Bias {i}: {biasWeights[i]}\n");
            }
            return s.ToString();
        }
    }
}