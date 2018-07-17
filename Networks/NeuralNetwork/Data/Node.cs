using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Data
{
    using System.Linq;

    public class Node
    {
        /// <summary>
        ///     The weights, with reference to the layer & node the value id being mapped from
        /// </summary>
        public Dictionary<Layer, Dictionary<Node, double>> Weights { get; set; }

        /// <summary>
        ///     The bias weights, with reference to the layer the value is mapped from
        /// </summary>
        public Dictionary<Layer, double> BiasWeights;

        /// <summary>
        ///     The output of the node from the last results calculation.
        /// </summary>
        public double Output { get; set; }

        public Node()
        {
            // default constructor
        }

        public Node(IReadOnlyList<Layer> nodeGroupPrev)
        {
            Weights = new Dictionary<Layer, Dictionary<Node, double>>();
            foreach (var prevNodeLayer in nodeGroupPrev)
            {
                var correspondingNodeWeights = new Dictionary<Node, double>();
                foreach (var node in prevNodeLayer.Nodes)
                {
                    correspondingNodeWeights.Add(node, 0);
                }
                Weights.Add(prevNodeLayer, correspondingNodeWeights);
            }

            BiasWeights = new Dictionary<Layer, double>();
            foreach (var prevNodeLayer in nodeGroupPrev)
            {
                BiasWeights.Add(prevNodeLayer, 0);
            }
        }

        public override string ToString()
        {
            var s = new StringBuilder();
            foreach (var layerWeightKey in Weights.Keys.ToList())
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