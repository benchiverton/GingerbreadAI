using System;
using System.Collections.Generic;
using System.Linq;
using Library.Computations;

namespace Model.NeuralNetwork.Models
{
    public class Node
    {
        /// <summary>
        ///     The weights, with reference to the layer & node the value id being mapped from
        /// </summary>
        public Dictionary<Node, Weight> Weights { get; set; } = new Dictionary<Node, Weight>();

        /// <summary>
        ///     The bias weights, with reference to the layer the value is mapped from
        /// </summary>
        public Dictionary<Layer, Weight> BiasWeights { get; set; } = new Dictionary<Layer, Weight>();

        /// <summary>
        ///     The output of the node from the last results calculation.
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

        public void CalculateOutput()
        {
            var output = Weights.Sum(previousNodeWeight => previousNodeWeight.Key.Output * previousNodeWeight.Value.Value) 
                         + BiasWeights.Sum(previousLayerWeight => previousLayerWeight.Value.Value);

            Output = LogisticFunction.ComputeOutput(output);
        }
    }
}