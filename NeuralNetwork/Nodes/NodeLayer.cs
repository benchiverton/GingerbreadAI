using System;
using System.Text;
using NeuralNetwork.Library;

namespace NeuralNetwork.Nodes
{
    /// <summary>
    ///     The NodeLayers are stored in a Linked List, so it needs to contain a reference to the NodeLayer before it.
    /// </summary>
    public class NodeLayer
    {
        /// <summary>
        ///     The name of the node (purely for helping you design and navagate your network).
        /// </summary>
        public string Name;

        /// <summary>
        ///     An array of the nodes within this layer.
        /// </summary>
        public Node[] Nodes;

        /// <summary>
        ///     An array containing the NodeLayers that feed into this one.
        /// </summary>
        public NodeLayer[] PreviousLayers;

        /// <summary>
        ///     Should be used to set up the input
        /// </summary>
        /// <param name="name"></param>
        /// <param name="nodeCount"></param>
        public NodeLayer(string name, int nodeCount)
        {
            Name = name;
            Nodes = new Node[nodeCount];
            PreviousLayers = null;
        }

        /// <summary>
        ///     Constructs a NodeLayer, initialising each node with the correct amount of Weights.
        /// </summary>
        /// <param name="name"></param>
        /// <param name="nodeCount"></param>
        /// <param name="previousLayers"></param>
        public NodeLayer(string name, int nodeCount, NodeLayer[] previousLayers)
        {
            Name = name;
            Nodes = new Node[nodeCount];
            for (var i = 0; i < nodeCount; i++)
                Nodes[i] = new Node(previousLayers);
            PreviousLayers = previousLayers;
        }

        /// <summary>
        ///     Initialises this NodeLayer with the parameters supplied.
        /// </summary>
        /// <param name="name"></param>
        /// <param name="nodes"></param>
        /// <param name="previousLayer"></param>
        public NodeLayer(string name, Node[] nodes, NodeLayer[] previousLayer)
        {
            Name = name;
            Nodes = nodes;
            PreviousLayers = previousLayer;
        }

        /// <summary>
        ///     Initialises each Node in Nodes with random weights.
        /// </summary>
        /// <param name="rand"></param>
        public void Initialise(Random rand)
        {
            foreach (var node in Nodes)
                node?.Initialise(rand);
        }

        /// <summary>
        ///     Returns the result from this nodeLayer, using its previous layers.
        /// </summary>
        /// <param name="inputs"></param>
        /// <returns></returns>
        public double[] GetResult(double[] inputs)
        {
            // this should only happen when you reach an input layer
            if (PreviousLayers == null)
                return inputs;
            // we have a result for each node, so I initialise the result array here
            var results = new double[Nodes.Length];
            // select a layer feeding into this one
            PreviousLayers.Each((layer, i) =>
            {
                // gets the outputs of a previous layer, which are the inputs for this layer
                var layerInputs = layer.GetResult(inputs);
                // iterate through Nodes in this layer
                for (var j = 0; j < Nodes.Length; j++)
                {
                    // iterate through the nodes of a previous layer, adding its weighted output to the results
                    for (var k = 0; k < layer.Nodes.Length; k++)
                        results[j] += layerInputs[k] * Nodes[j].Weights[i][k];

                    // add the bias for the previous layer
                    results[j] += Nodes[j].BiasWeights[i];
                }
            });

            // apply the logistic function to each of the results
            for (var i = 0; i < results.Length; i++)
                results[i] = Calculations.LogisticFunction(results[i]);

            return results;
        }

        public override string ToString()
        {
            var s = new StringBuilder($"Node Layer: {Name}\n");
            for (var i = 0; i < Nodes.Length; i++)
                s.Append($"Node {i}:\n{Nodes[i]}");
            if (PreviousLayers != null)
            {
                s.Append("Previous Layers:\n");
                foreach (var nodeLayer in PreviousLayers)
                    s.Append($"{nodeLayer.Name}\n");
            }
            s.Append("----------\n");
            return s.ToString();
        }
    }
}