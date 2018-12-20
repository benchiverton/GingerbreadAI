using System;
using System.Collections.Generic;

namespace NeuralNetwork.Data
{
    [Serializable]
    public class Node
    {
        /// <summary>
        ///     The weights, with reference to the layer & node the value id being mapped from
        /// </summary>
        public Dictionary<Node, Weight> Weights { get; set; }

        /// <summary>
        ///     The bias weights, with reference to the layer the value is mapped from
        /// </summary>
        public Dictionary<Layer, Weight> BiasWeights;

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
            Weights = new Dictionary<Node, Weight>();
            foreach (var prevNodeLayer in nodeGroupPrev)
            {
                foreach (var node in prevNodeLayer.Nodes)
                {
                    Weights.Add(node, new Weight(0));
                }
            }

            BiasWeights = new Dictionary<Layer, Weight>();
            foreach (var prevNodeLayer in nodeGroupPrev)
            {
                BiasWeights.Add(prevNodeLayer, new Weight(0));
            }
        }

        public Node(Dictionary<Node, Weight> weights, Dictionary<Layer, Weight> biasWeights, double output)
        {
            Weights = weights;
            BiasWeights = biasWeights;
            Output = output;
        }
    }
}