using System;
using System.Collections.Generic;

namespace GingerbreadAI.Model.NeuralNetwork.Models
{
    public class Node
    {
        /// <summary>
        /// The weights, with reference to the layer & node the value id being mapped from
        /// </summary>
        public Dictionary<Node, Weight> Weights { get; set; } = new Dictionary<Node, Weight>();

        /// <summary>
        /// The bias weights, with reference to the layer the value is mapped from
        /// </summary>
        public Dictionary<Layer, Weight> BiasWeights { get; set; } = new Dictionary<Layer, Weight>();

        /// <summary>
        /// The output of the node from the last results calculation.
        /// </summary>
        public double Output { get; set; }

        public Node()
        {
        }

        public Node(IReadOnlyList<Layer> nodeGroupPrev)
        {
            foreach (var prevNodeLayer in nodeGroupPrev)
            {
                foreach (var node in prevNodeLayer.Nodes)
                {
                    Weights.Add(node, new Weight(0));
                }
            }

            foreach (var prevNodeLayer in nodeGroupPrev)
            {
                BiasWeights.Add(prevNodeLayer, new Weight(0));
            }
        }

        public void CalculateOutput(Func<double, double> activationFunction)
        {
            var output = 0d;

            // TODO: optimise this
            foreach (var weight in Weights)
            {
                output += weight.Key.Output * weight.Value.Value;
            }
            foreach (var weight in BiasWeights)
            {
                output += weight.Value.Value;
            }

            Output = activationFunction.Invoke(output);
        }
    }
}