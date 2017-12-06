using System;
using System.Text;

namespace NeuralNetwork.Nodes
{
    /// <summary>
    /// A class containing the properties and methods that a single node in a network requires.
    /// </summary>
    public class Node
    {
        /// <summary>
        /// The weights associated with a node. These values correspond to the nodeLayer which 
        /// feed into the nodeLayer containing this node, ie Weights[1][0] => nodeLayerPrev[1].Nodes[0].
        /// </summary>
        public double[][] Weights { get; set; }
        /// <summary>
        /// The bias that is passed into this node from the previous layerSSS!!
        /// </summary>
        public double[] BiasWeights { get; set; }

        /// <summary>
        /// Constructs a node with the supplied weights & bias.
        /// </summary>
        /// <param name="weights"></param>
        /// <param name="biasWeights"></param>
        public Node(double[][] weights, double[] biasWeights)
        {
            Weights = weights;
            BiasWeights = biasWeights;
        }

        /// <summary>
        /// Constructs a node with the correct amount of weights, given an array of the NodeLayers which feed into this node.
        /// </summary>
        /// <param name="nodeLayerPrev"></param>
        public Node(NodeLayer[] nodeLayerPrev)
        {
            Weights = new double[nodeLayerPrev.Length][];
            // each double[] in weights corresponds to the relevant array of nodes in the previous layer.
            for (var i = 0; i < nodeLayerPrev.Length; i++)
            {
                Weights[i] = new double[nodeLayerPrev[i].Nodes.Length];
            }
            BiasWeights = new double[nodeLayerPrev.Length];
        }

        /// <summary>
        /// Initialises this Node with random weights.
        /// </summary>
        /// <param name="rand"></param>
        public void Initialise(Random rand)
        {
            foreach (var weightArr in Weights)
            {
                for (var j = 0; j < weightArr.Length; j++)
                {
                    weightArr[j] = (double)rand.Next(1000000) / 1000000;
                }
            }
            for (int i=0; i<BiasWeights.Length; i++)
            {
                BiasWeights[i] = (double) rand.Next(1000000) / 1000000;
            }
        }

        public override string ToString()
        {
            var s = new StringBuilder();
            for (var i = 0; i < Weights.Length; i++)
            {
                s.Append($"Node Array {i}:\n");
                for (var j = 0; j < Weights[i].Length; j++)
                {
                    s.Append($"Weight {j}: {Weights[i][j]}\n");
                }
            }
            for (var i = 0; i < BiasWeights.Length; i++)
            {
                s.Append($"Bias {i}: {BiasWeights[i]}\n");
            }
            return s.ToString();
        }
    }
}
