using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Nodes
{
    /// <summary>
    /// A class containing the properties and methods that a single node in a network requires.
    /// </summary>
    class Node
    {
        /// <summary>
        /// The weights associated with a node. These values correspond to the nodeLayer which 
        /// feed into the nodeLayer containing this node, ie Weights[0] => nodeLayerPrev.Nodes[0].
        /// </summary>
        public double[] Weights { get; set; }

        // should be randomised if no previous moel is defined, although this should be done to the array parsed.
        public Node(double[] weights)
        {
            Weights = weights;
        }
    }
}
