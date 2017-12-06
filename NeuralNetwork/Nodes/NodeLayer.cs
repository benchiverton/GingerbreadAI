using System;
using System.Text;
using NeuralNetwork.Library;

namespace NeuralNetwork.Nodes
{
    /// <summary>
    /// The NodeLayers are stored in a Linked List, so it needs to contain a reference to the NodeLayer before it.
    /// </summary>
    public class NodeLayer
    {
        /// <summary>
        /// The name of the node (purely for helping you design and navagate your network).
        /// </summary>
        public string Name;
        /// <summary>
        /// An array of the nodes within this layer.
        /// </summary>
        public Node[] Nodes;
        /// <summary>
        /// An array containing the NodeLayers that feed into this one.
        /// </summary>
        public NodeLayer[] PreviousLayers;

        /// <summary>
        /// Should be used to set up the input
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
        /// Constructs a NodeLayer, initialising each node with the correct amount of Weights.
        /// </summary>
        /// <param name="name"></param>
        /// <param name="nodeCount"></param>
        /// <param name="previousLayers"></param>
        public NodeLayer(string name, int nodeCount, NodeLayer[] previousLayers)
        {
            Name = name;
            Nodes = new Node[nodeCount];
            for (var i = 0; i < nodeCount; i++)
            {
                Nodes[i] = new Node(previousLayers);
            }
            PreviousLayers = previousLayers;
        }

        /// <summary>
        /// Initialises this NodeLayer with the parameters supplied.
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
        /// Initialises each Node in Nodes with random weights.
        /// </summary>
        /// <param name="rand"></param>
        public void Initialise(Random rand)
        {
            foreach (var node in Nodes)
            {
                node?.Initialise(rand);
            }
        }

        /// <summary>
        /// Returns the result from this nodeLayer, using its previous layers.
        /// </summary>
        /// <param name="inputs"></param>
        /// <returns></returns>
        public double[] GetResult(double[] inputs)
        {
            if (PreviousLayers == null)
            {
                return inputs;
            }
            var results = new double[Nodes.Length];
            for (var i = 0; i < results.Length; i++)
            {
                results[i] = 0;
            }
            // select a layer feeding into this one
            for (var i = 0; i < PreviousLayers.Length; i++)
            {
                var layerInputs = PreviousLayers[i].GetResult(inputs);
                // select a node from the previous layer
                for (var j = 0; j < PreviousLayers[i].Nodes.Length; j++)
                {
                    // select a node from this layer
                    for (var k = 0; k < Nodes.Length; k++)
                    {
                        results[k] += layerInputs[j] * Nodes[k].Weights[i][j];
                    }
                }
                // adding the bias
                for (var j = 0; j < Nodes.Length; j++)
                {
                    results[j] += Nodes[j].BiasWeights[i];
                }
            }
            for (var i = 0; i < results.Length; i++)
            {
                results[i] = Calculations.LogisticFunction(results[i]);
            }
            return results;
        }

        public override string ToString()
        {
            var s = new StringBuilder($"Node Layer: {Name}\n");
            for(var i=0; i<Nodes.Length; i++)
            {
                s.Append($"Node {i}:\n{Nodes[i]}");
            }
            if (PreviousLayers != null)
            {
                s.Append("Previous Layers:\n");
                foreach (var nodeLayer in PreviousLayers)
                {
                    s.Append($"{nodeLayer.Name}\n");
                }
            }
            s.Append("----------\n");
            return s.ToString();
        }
    }
}
