using AI.Calculations;
using System;
using System.Collections.Generic;

namespace NeuralNetwork.Models
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
        public Dictionary<Layer, Weight> BiasWeights { get; set; }

        /// <summary>
        ///     The output of the node from the last results calculation.
        /// </summary>
        public double Output { get; set; } = 0;

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

        public void PopulateOutput()
        {
            var output = 0d;
            foreach (var previousNodeWeight in Weights)
            {
                output += previousNodeWeight.Key.Output * previousNodeWeight.Value.Value;
            }
            foreach (var previousLayerWeight in BiasWeights)
            {
                output += previousLayerWeight.Value.Value;
            }

            Output = NetworkCalculations.LogisticFunction(output);
        }
    }
}